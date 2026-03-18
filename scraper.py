
from firecrawl import FirecrawlApp

TARGET_URL = "https://www.tawuniya.com/"
FIRECRAWL_API_KEY = "fc-97ca75d61c5e40759585f87e7f49dfc4"



def scrape_with_firecrawl(output_file="firecrawl.txt"):
    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

    print("Crawling with Firecrawl...")
    result = app.crawl(
        TARGET_URL,
        limit=500,
        allow_subdomains=False,
    )

    pages = result.data if hasattr(result, "data") else []
    print(f"Firecrawl crawled {len(pages)} pages.")

    with open(output_file, "w", encoding="utf-8") as f:
        for page in pages:
            url = (page.metadata.url if page.metadata and page.metadata.url else "")
            content = page.markdown or ""
            f.write(f"=== URL: {url} ===\n")
            f.write(content)
            f.write("\n\n")

    print(f"Firecrawl output saved to {output_file}")


if __name__ == "__main__":
    scrape_with_firecrawl()
