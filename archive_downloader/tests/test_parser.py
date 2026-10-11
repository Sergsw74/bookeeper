"""
Tests for arXiv HTML parser and pagination discovery.
"""

from archive_downloader.parser import parse_arxiv_list_page, sanitize_filename


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<body>
<div id="dlpage">
  <div class="paging">
    Total of 120 entries :
    <span>1-50</span>
    <a href="/list/cs.AI/recent?skip=50&amp;show=50">51-100</a>
    <a href="/list/cs.AI/recent?skip=100&amp;show=50">101-120</a>
  </div>
  <dl id="articles">
    <dt>
      <a name="item1">[1]</a>
      <a href="/abs/2610.12345" title="Abstract">arXiv:2610.12345</a>
      [<a href="/pdf/2610.12345" title="Download PDF">pdf</a>, <a href="/html/2610.12345">html</a>]
    </dt>
    <dd>
      <div class="meta">
        <div class="list-title mathjax">
          <span class="descriptor">Title:</span> Agentic AI: Autonomous Reasoning Architecture
        </div>
        <div class="list-authors">
          <a href="/search/?query=Alice+Smith">Alice Smith</a>,
          <a href="/search/?query=Bob+Jones">Bob Jones</a>
        </div>
        <div class="list-subjects">
          <span class="descriptor">Subjects:</span>
          <span class="primary-subject">Artificial Intelligence (cs.AI)</span>; Computation and Language (cs.CL)
        </div>
      </div>
    </dd>
    <dt>
      <a name="item2">[2]</a>
      <a href="/abs/2610.12346" title="Abstract">arXiv:2610.12346</a>
      [<a href="/pdf/2610.12346" title="Download PDF">pdf</a>]
    </dt>
    <dd>
      <div class="meta">
        <div class="list-title mathjax">
          <span class="descriptor">Title:</span> Multi-Modal World Models for Robotics
        </div>
        <div class="list-authors">
          <a href="/search/?query=Charlie+Brown">Charlie Brown</a>
        </div>
        <div class="list-subjects">
          <span class="descriptor">Subjects:</span>
          <span class="primary-subject">Robotics (cs.RO)</span>
        </div>
      </div>
    </dd>
  </dl>
</div>
</body>
</html>
"""


def test_sanitize_filename():
    assert sanitize_filename("A/B\\C: D*E? F\"G<H>I|J") == "A_B_C_ D_E_ F_G_H_I_J"
    assert sanitize_filename("Normal Title") == "Normal Title"
    long_title = "a" * 200
    assert len(sanitize_filename(long_title, max_length=50)) <= 50


def test_parse_arxiv_list_page():
    articles, next_url = parse_arxiv_list_page(
        SAMPLE_HTML,
        current_url="https://arxiv.org/list/cs.AI/recent",
    )

    assert len(articles) == 2

    # Article 1
    a1 = articles[0]
    assert a1.arxiv_id == "2610.12345"
    assert a1.title == "Agentic AI: Autonomous Reasoning Architecture"
    assert a1.authors == ["Alice Smith", "Bob Jones"]
    assert "cs.AI" in a1.primary_category
    assert a1.pdf_url == "https://arxiv.org/pdf/2610.12345"
    assert "pdf" in a1.available_formats

    # Article 2
    a2 = articles[1]
    assert a2.arxiv_id == "2610.12346"
    assert a2.title == "Multi-Modal World Models for Robotics"
    assert a2.authors == ["Charlie Brown"]
    assert a2.pdf_url == "https://arxiv.org/pdf/2610.12346"

    # Pagination
    assert next_url == "https://arxiv.org/list/cs.AI/recent?skip=50&show=50"


def test_parse_arxiv_last_page():
    last_page_html = """
    <html><body>
      <div class="paging">
        <span>101-120</span>
      </div>
      <dl>
        <dt><a href="/abs/2610.99999">arXiv:2610.99999</a></dt>
        <dd><div class="list-title">Final Article</div></dd>
      </dl>
    </body></html>
    """
    articles, next_url = parse_arxiv_list_page(
        last_page_html,
        current_url="https://arxiv.org/list/cs.AI/recent?skip=100&show=50",
    )
    assert len(articles) == 1
    assert next_url is None
