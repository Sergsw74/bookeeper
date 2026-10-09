"""Unit tests for Wake-on-LAN (WOL) integration across Ollama pool and warmup flow."""

import socket
from unittest.mock import MagicMock, patch

import pytest

from bookeeper.config import OllamaServerConfig
from bookeeper.processing.extractor import KnowledgeExtractor
from bookeeper.processing.ollama_pool import OllamaPool, OllamaServerNode, send_wake_on_lan


def test_send_wake_on_lan_packet_structure():
    """Verify standard AMD Magic Packet generation and broadcast socket options."""
    test_mac = "D8:43:AE:FA:44:95"
    expected_bytes = b"\xff" * 6 + bytes.fromhex("D843AEFA4495") * 16
    assert len(expected_bytes) == 102

    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_sock_cls.return_value = mock_sock

        res = send_wake_on_lan(test_mac, broadcast_ip="192.168.50.255", port=9)
        assert res is True
        mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        mock_sock.sendto.assert_called_once_with(expected_bytes, ("192.168.50.255", 9))


def test_send_wake_on_lan_invalid_mac():
    """Verify ValueError is raised for invalid MAC formats."""
    with pytest.raises(ValueError, match="Invalid MAC address"):
        send_wake_on_lan("invalid-mac")

    with pytest.raises(ValueError, match="Invalid MAC address"):
        send_wake_on_lan("00:11:22:33")


def test_ollama_server_config_mac_normalization():
    """Verify MAC address parsing and normalization in OllamaServerConfig."""
    cfg1 = OllamaServerConfig(url="http://192.168.50.118:11434", mac="d8-43-ae-fa-44-95")
    assert cfg1.mac == "D8:43:AE:FA:44:95"

    # Test wol_mac alias
    cfg2 = OllamaServerConfig.model_validate({
        "url": "http://192.168.50.118:11434",
        "wol_mac": "d843aefa4495",
    })
    assert cfg2.mac == "D8:43:AE:FA:44:95"


def test_ollama_server_node_wake():
    """Verify OllamaServerNode.wake() triggers send_wake_on_lan."""
    node = OllamaServerNode(
        url="http://192.168.50.118:11434",
        mac="D8:43:AE:FA:44:95",
        wol_broadcast="255.255.255.255",
        wol_port=9,
        wol_secret="01:02:03:04:05:06",
    )
    with patch("bookeeper.processing.ollama_pool.send_wake_on_lan", return_value=True) as mock_wol:
        assert node.wake() is True
        mock_wol.assert_called_once_with(
            "D8:43:AE:FA:44:95",
            broadcast_ip="255.255.255.255",
            port=9,
            secret="01:02:03:04:05:06",
        )

    node_no_mac = OllamaServerNode(url="http://192.168.50.15:11434")
    assert node_no_mac.wake() is False


def test_send_wake_on_lan_with_hex_secret():
    """Verify 6-byte hex SecureOn password appends correctly (108 bytes total)."""
    test_mac = "D8:43:AE:FA:44:95"
    secret = "01:02:03:04:05:06"
    expected_bytes = (
        b"\xff" * 6
        + bytes.fromhex("D843AEFA4495") * 16
        + bytes.fromhex("010203040506")
    )
    assert len(expected_bytes) == 108

    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_sock_cls.return_value = mock_sock

        res = send_wake_on_lan(test_mac, secret=secret)
        assert res is True
        mock_sock.sendto.assert_called_once_with(expected_bytes, ("255.255.255.255", 9))


def test_send_wake_on_lan_with_ip_secret():
    """Verify 4-byte IP address secret appends correctly (106 bytes total)."""
    test_mac = "D8:43:AE:FA:44:95"
    secret = "192.168.50.118"
    expected_bytes = (
        b"\xff" * 6
        + bytes.fromhex("D843AEFA4495") * 16
        + socket.inet_aton("192.168.50.118")
    )
    assert len(expected_bytes) == 106

    with patch("socket.socket") as mock_sock_cls:
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_sock_cls.return_value = mock_sock

        res = send_wake_on_lan(test_mac, secret=secret)
        assert res is True
        mock_sock.sendto.assert_called_once_with(expected_bytes, ("255.255.255.255", 9))


def test_ollama_server_config_secret_aliases():
    """Verify wol_secret and aliases (secret, wol_password) in OllamaServerConfig."""
    cfg1 = OllamaServerConfig.model_validate({
        "url": "http://192.168.50.118:11434",
        "mac": "D8:43:AE:FA:44:95",
        "wol_secret": "01:02:03:04:05:06",
    })
    assert cfg1.wol_secret == "01:02:03:04:05:06"

    cfg2 = OllamaServerConfig.model_validate({
        "url": "http://192.168.50.118:11434",
        "mac": "D8:43:AE:FA:44:95",
        "secret": "010203040506",
    })
    assert cfg2.wol_secret == "010203040506"

    cfg3 = OllamaServerConfig.model_validate({
        "url": "http://192.168.50.118:11434",
        "mac": "D8:43:AE:FA:44:95",
        "wol_password": "mysecret",
    })
    assert cfg3.wol_secret == "mysecret"


def test_warmup_wol_triggers_and_retries():
    """Verify warmup sends WOL to unreachable node and re-probes upon boot."""
    server1 = OllamaServerConfig(url="http://192.168.50.118:11434", priority=1, name="sleeping-gpu", mac="D8:43:AE:FA:44:95")
    server2 = OllamaServerConfig(url="http://192.168.50.15:11434", priority=2, name="active-server")
    pool = OllamaPool(servers=[server1, server2])

    extractor = KnowledgeExtractor(
        pool=pool,
        model="qwen2.5:3b",
        wol_enabled=True,
        wol_wait_seconds=2,
        wol_probe_interval=0.1,
    )

    status_messages = []
    def status_callback(msg):
        status_messages.append(msg)

    extractor_node1 = pool.nodes[0]
    with patch.object(extractor_node1, "wake", wraps=extractor_node1.wake):
        with patch("bookeeper.processing.ollama_pool.send_wake_on_lan", return_value=True) as mock_wol:
            with patch("time.sleep", return_value=None):
                with patch("urllib.request.urlopen") as mock_open:
                    probe_count = [0]
                    def custom_urlopen(req, timeout=15):
                        url = req.full_url
                        if "192.168.50.118" in url:
                            probe_count[0] += 1
                            if probe_count[0] <= 2:
                                raise ConnectionError("Host down")
                            # After WOL packet, probe succeeds!
                            m = MagicMock()
                            if "/api/ps" in url:
                                m.read.return_value = b'{"models": [{"name": "qwen2.5:3b", "size": 1000, "size_vram": 1000, "runner": "cuda"}]}'
                            else:
                                m.read.return_value = b'{"version": "0.4.0"}'
                            m.__enter__.return_value = m
                            return m
                        else:
                            m = MagicMock()
                            m.read.return_value = b'{"models": [{"name": "qwen2.5:3b", "size": 1000, "size_vram": 1000, "runner": "cuda"}]}'
                            m.__enter__.return_value = m
                            return m

                    mock_open.side_effect = custom_urlopen
                    res = extractor.warmup_and_check_device(on_status_callback=status_callback)

                    assert mock_wol.called
                    assert res["status"] == "ok"
                    # Node 1 woke up and became active GPU primary
                    assert res["primary"]["url"] == "http://192.168.50.118:11434"
                    assert res["primary"]["status"] == "ok"
                    assert res["primary"]["is_gpu"] is True
                    assert any("Woke up" in m or "awake" in m for m in status_messages)


def test_warmup_wol_timeout_graceful_fallback():
    """Verify that if woken server never responds within timeout, warmup completes gracefully."""
    server1 = OllamaServerConfig(url="http://192.168.50.118:11434", priority=1, name="dead-gpu", mac="D8:43:AE:FA:44:95")
    server2 = OllamaServerConfig(url="http://192.168.50.15:11434", priority=2, name="backup-server")
    pool = OllamaPool(servers=[server1, server2])

    extractor = KnowledgeExtractor(
        pool=pool,
        model="qwen2.5:3b",
        wol_enabled=True,
        wol_wait_seconds=0.2,
        wol_probe_interval=0.1,
    )

    status_messages = []
    def status_callback(msg):
        status_messages.append(msg)

    with patch("bookeeper.processing.ollama_pool.send_wake_on_lan", return_value=True) as mock_wol:
        with patch("time.sleep", return_value=None):
            with patch("urllib.request.urlopen") as mock_open:
                def custom_urlopen(req, timeout=15):
                    url = req.full_url
                    if "192.168.50.118" in url:
                        raise ConnectionError("Host dead")
                    m = MagicMock()
                    m.read.return_value = b'{"models": [{"name": "qwen2.5:3b", "size": 1000, "size_vram": 1000, "runner": "cuda"}]}'
                    m.__enter__.return_value = m
                    return m

                mock_open.side_effect = custom_urlopen
                res = extractor.warmup_and_check_device(on_status_callback=status_callback)

                assert mock_wol.called
                assert res["status"] == "ok"
                # Server 1 remains unreachable, Server 2 is selected as active primary
                server1_rep = next(r for r in res["servers"] if r["url"] == "http://192.168.50.118:11434")
                assert server1_rep["status"] == "unreachable"
                assert res["primary"]["url"] == "http://192.168.50.15:11434"
                assert any("did not respond" in m for m in status_messages)
