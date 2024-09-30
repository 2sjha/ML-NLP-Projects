"""
Web Crawler to create the knowledge base for the chatbot
"""

from typing import List
from requests import RequestException, get
from bs4 import BeautifulSoup
from utils import Page, CrawledData, get_domain, match
from filters import INITIAL_LINKS, BLOCK_LIST, SAME_DOMAIN_BLOCK_LIST


def crawl(pages: List[Page]) -> List[CrawledData]:
    """
    Crawls the links and reads the text info in that webpage
    Returns the Text from links and Set of forward links
    """
    crawled_data = []
    for page in pages:
        crawled_links = set()
        page_domain = get_domain(page.url)
        response = False
        try:
            response = get(
                page.url, timeout=5  # timeout after 5 seconds of response not received
            )
        except RequestException as _e:
            print("Couldn't read URL", page.url)

        if not response:
            continue

        soup = BeautifulSoup(response.content, "html.parser")

        page_text = ""
        same_domain_links = 0

        # Read the text from the page
        for p in soup.select("p"):
            page_text = page_text + "\n" + p.get_text()

        # Crawl for more links only for the initial links list
        if page.crawl_links:
            # Look for links in this page to add to our crawl list
            # Avoid visiting pages that are in the block list
            for crawled_link in soup.find_all("a"):
                href = crawled_link.get("href")
                if href and href.startswith("http"):
                    href_domain = get_domain(href)
                    if (
                        href_domain not in BLOCK_LIST
                        and href != page.url
                        and not match(href, SAME_DOMAIN_BLOCK_LIST)
                    ):
                        # Add this link to in the crawl set If the crawled link
                        # does not have the same domain as the original link
                        # Or if it has then we limit same domain crawls
                        if href_domain != page_domain:
                            crawled_links.add(href)
                            print("Found link:", href)
                        elif same_domain_links < 5:
                            crawled_links.add(href)
                            print("Found link:", href)
                            same_domain_links += 1

        crawled_data.append(CrawledData(page.url, page_text, crawled_links))

    return crawled_data


def start_crawling():
    """
    Starts crawling initial links for text and more links
    Then We dont want to crawl further, so we just read the text from crawled links
    """
    crawled_data = crawl(INITIAL_LINKS)
    more_links = []
    for page_data in crawled_data:
        for link in page_data.crawled_links:
            more_links.append(Page(link, False))

    crawled_data = crawled_data + crawl(more_links)

    for idx, page_data in enumerate(crawled_data):
        with open(
            "crawled_files/crawl_" + str(idx) + ".txt", "w", encoding="utf8"
        ) as f:
            f.write(page_data.url + "\n" + page_data.text)


if __name__ == "__main__":
    start_crawling()
