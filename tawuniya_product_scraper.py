# import os
# import glob
# import re
# import json
# import time
# from firecrawl import Firecrawl
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY = os.getenv("FIRECRAWL_API_KEY", "YOUR_API_KEY_HERE")

# def scrape_tawuniya_products():
#     if API_KEY == "YOUR_API_KEY_HERE":
#         print("⚠️  Warning: Please replace 'YOUR_API_KEY_HERE' with your actual Firecrawl API key.")
#         print("You can also set it in a .env file: FIRECRAWL_API_KEY=your_key")
#         return

#     print("Initializing Firecrawl...")
#     app = Firecrawl(api_key=API_KEY)  # ✅ v2: Firecrawl, not FirecrawlApp
    
#     import glob

#     # The starting URL to crawl
#     target_url = "https://www.tawuniya.com/ar"
#     specific_urls = []
#     # --- RESUME LOGIC ---
#     # Find all previously created files
#     existing_files = glob.glob("tawuniya_products_data*.json")
    
#     # Determine the next file number
#     file_count = len(existing_files)
#     if file_count == 0:
#         output_file = "tawuniya_products_data_1.json"
#     else:
#         output_file = f"tawuniya_products_data_{file_count + 1}.json"

#     # Load previously scraped URLs so we don't scrape them again!
#     existing_urls = []
    
#     for file_name in existing_files:
#         try:
#             with open(file_name, 'r', encoding='utf-8') as f:
#                 past_data = json.load(f)
#                 for item in past_data:
#                     url = item.get("source_url")
#                     # We collect the URLs to exclude, but we MUST NOT exclude our root target_url
#                     if url and url != target_url and url not in existing_urls:
#                         existing_urls.append(url)
#         except Exception as e:
#             print(f"⚠️ Could not load data from {file_name}. Error: {e}")

#     print(f"📖 Loaded {len(existing_urls)} previously scraped URLs from {len(existing_files)} history files.")
    
#     new_scraped_data = []

#     # We will limit the crawl to avoid massive billing limits during testing (e.g. 50 pages).
#     # You can remove limit=50 or increase it to crawl the whole entire site.
#     print(f"Starting to crawl the website starting from: {target_url}...")
#     if existing_urls:
#         print(f"⏭️ Skipping {len(existing_urls)} pages we already have.")
    
#     try:
#         # ✅ v2: app.crawl() recursively finds pages.
#         # Pass existing_urls into exclude_paths so we "resume" scraping!
#         crawl_result = app.crawl(
#             target_url,
#             limit=50, # Optional: Limit to 50 pages to start with
#             exclude_paths=existing_urls if existing_urls else None,
#             scrape_options={
#                 "formats": ["markdown"],
#                 "excludeTags": ["img", "svg", "icon", "i", "picture", "figure", "nav", "footer"]
#             }
#         )

#         # The new pages found in this run
#         new_pages = getattr(crawl_result, "data", [])
        
#         if not new_pages:
#              # Just in case crawl_result is a dict or some other structure in the SDK
#             if isinstance(crawl_result, dict):
#                 new_pages = crawl_result.get("data", [])
#             elif hasattr(crawl_result, "items"):
#                 new_pages = crawl_result.items # some versions do this
                
#         print(f"✅ Crawl completed! Found {len(new_pages)} NEW pages.")

#         for index, page in enumerate(new_pages, 1):
#             # For each page the attributes can be accessed like object attributes or dictionary keys
#             # Let's extract metadata and markdown safely
#             metadata = getattr(page, "metadata", {})
#             if isinstance(page, dict):
#                 metadata = page.get("metadata", {})
#                 markdown_text = page.get("markdown", "")
#             else:
#                 markdown_text = getattr(page, "markdown", "")

#             # Try to get the URL from metadata or page directly
#             source_url = ""
#             if isinstance(metadata, dict):
#                 source_url = metadata.get("source_url") or metadata.get("sourceURL") or metadata.get("url")
#                 title = metadata.get("title", 'No Title')
#                 desc = metadata.get("description", '')
#             else:
#                 source_url = getattr(metadata, "source_url", "") or getattr(metadata, "sourceURL", "") or getattr(metadata, "url", "")
#                 title = getattr(metadata, "title", "No Title")
#                 desc = getattr(metadata, "description", "")

#             # Fallbacks if it is directly on the page object
#             if not source_url:
#                 if isinstance(page, dict):
#                     source_url = page.get("url") or page.get("source_url") or ""
#                 else:
#                     source_url = getattr(page, "url", "") or getattr(page, "source_url", "")
            
#             content = markdown_text or ""
#             # Clean up the exact same way as before
#             # 1. Remove markdown images entirely (e.g. ![AltText](URL))
#             content = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', content)
            
#             # 2. Remove markdown links but keep the text for cleaner RAG chunking
#             content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
            
#             # 3. Remove bare http/https URLs that weren't caught by the markdown link regex
#             content = re.sub(r'https?://[^\s]+', '', content)
            
#             # 4. Remove leftover standalone `!` marks that were attached to orphaned text (like !icon)
#             content = re.sub(r'![a-zA-Z0-9_-]+', '', content)
            
#             page_data = {
#                 "source_url": source_url,
#                 "title": title,
#                 "description": desc,
#                 "content_markdown": content,
#             }

#             new_scraped_data.append(page_data)

#         # Save to JSON
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(new_scraped_data, f, ensure_ascii=False, indent=2)
            
#         print(f"\n🎉 Process complete! We now have {len(new_scraped_data)} new pages saved.")
#         print(f"💾 Data saved to: {output_file}")

#     except Exception as e:
#         print(f"❌ Failed to crawl {target_url}. Error: {e}")

# if __name__ == "__main__":
#     scrape_tawuniya_products()

import os
import glob
import re
import json
from firecrawl import Firecrawl
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FIRECRAWL_API_KEY", "YOUR_API_KEY_HERE")
TARGET_URL = "https://www.tawuniya.com/ar"
PAGES_PER_FILE = 50
STATE_FILE = "scrape_state.json"  # Tracks progress across runs


# ─── State helpers ────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load the persistent state file that tracks all scraped URLs and file count."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"scraped_urls": [], "file_count": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ─── Content cleaner ──────────────────────────────────────────────────────────

def clean_content(raw: str) -> str:
    text = raw or ""
    text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', text)          # markdown images
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)      # markdown links → keep text
    text = re.sub(r'https?://[^\s]+', '', text)                  # bare URLs
    text = re.sub(r'![a-zA-Z0-9_-]+', '', text)                 # orphaned !icon
    text = re.sub(r'\n{3,}', '\n\n', text).strip()              # extra blank lines
    return text


# ─── Page extractor ───────────────────────────────────────────────────────────

def extract_page(page) -> dict:
    """Normalise a crawl result page regardless of SDK return type."""
    if isinstance(page, dict):
        metadata     = page.get("metadata", {})
        markdown_text = page.get("markdown", "")
    else:
        metadata     = getattr(page, "metadata", {}) or {}
        markdown_text = getattr(page, "markdown", "") or ""

    # --- URL ---
    if isinstance(metadata, dict):
        source_url = (metadata.get("source_url")
                      or metadata.get("sourceURL")
                      or metadata.get("url", ""))
        title = metadata.get("title", "No Title")
        desc  = metadata.get("description", "")
    else:
        source_url = (getattr(metadata, "source_url", "")
                      or getattr(metadata, "sourceURL", "")
                      or getattr(metadata, "url", ""))
        title = getattr(metadata, "title", "No Title")
        desc  = getattr(metadata, "description", "")

    if not source_url:
        source_url = (page.get("url", "") if isinstance(page, dict)
                      else getattr(page, "url", "") or getattr(page, "source_url", ""))

    return {
        "source_url":       source_url,
        "title":            title,
        "description":      desc,
        "content_markdown": clean_content(markdown_text),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def scrape_tawuniya_products():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  Set your FIRECRAWL_API_KEY in a .env file or environment variable.")
        return

    state = load_state()
    already_scraped: list = state["scraped_urls"]   # all URLs ever scraped
    file_count: int       = state["file_count"]

    print(f"📖 Resuming: {len(already_scraped)} pages already scraped across {file_count} file(s).")

    print("🔌 Initialising Firecrawl …")
    app = Firecrawl(api_key=API_KEY)

    # ── Crawl: ask for more than we need so we have room to skip duplicates ──
    # We request up to 3× our target so even if many are duplicates we can
    # still fill a full batch of 50 new pages.
    fetch_limit = PAGES_PER_FILE * 3

    print(f"🌐 Crawling {TARGET_URL} (fetching up to {fetch_limit} candidates) …")
    print(f"⏭️  Will skip {len(already_scraped)} previously scraped URLs.")

    try:
        crawl_result = app.crawl(
            TARGET_URL,
            limit=fetch_limit,
            exclude_paths=already_scraped if already_scraped else None,
            scrape_options={
                "formats": ["markdown"],
                "excludeTags": ["img", "svg", "icon", "i", "picture", "figure", "nav", "footer"]
            }
        )
    except Exception as e:
        print(f"❌ Crawl failed: {e}")
        return

    # Normalise result to a plain list
    if hasattr(crawl_result, "data"):
        raw_pages = crawl_result.data or []
    elif isinstance(crawl_result, dict):
        raw_pages = crawl_result.get("data", [])
    else:
        raw_pages = list(crawl_result) if crawl_result else []

    print(f"✅ Crawl returned {len(raw_pages)} candidate page(s).")

    # ── Deduplicate against state ─────────────────────────────────────────────
    already_set  = set(already_scraped)
    new_pages    = []
    seen_this_run = set()

    for page in raw_pages:
        record = extract_page(page)
        url    = record["source_url"]

        if not url:
            continue
        if url in already_set or url in seen_this_run:
            continue

        new_pages.append(record)
        seen_this_run.add(url)

        if len(new_pages) == PAGES_PER_FILE:
            break   # we have exactly 50 new pages — stop

    if not new_pages:
        print("🏁 No new pages found. The whole site may already be scraped!")
        return

    # ── Save the batch ────────────────────────────────────────────────────────
    file_count += 1
    output_file = f"tawuniya_products_data_{file_count}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_pages, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Saved {len(new_pages)} new pages → {output_file}")

    # ── Persist updated state ─────────────────────────────────────────────────
    state["scraped_urls"] = already_scraped + [p["source_url"] for p in new_pages]
    state["file_count"]   = file_count
    save_state(state)

    print(f"💾 State updated. Total scraped so far: {len(state['scraped_urls'])} pages in {file_count} file(s).")
    print(f"▶️  Run the script again to scrape the next {PAGES_PER_FILE} pages.")


if __name__ == "__main__":
    scrape_tawuniya_products()