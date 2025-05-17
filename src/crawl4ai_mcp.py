"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from playwright.async_api import Error as PlaywrightError
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import time
from functools import wraps

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_supabase_client, add_documents_to_supabase, search_documents

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    last_health_check: float = 0.0
    health_check_interval: float = 300.0  # Check browser health every 5 minutes
    
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration with minimal settings
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    crawler = None
    supabase_client = None
    
    try:
        # Initialize Supabase client first since it's less likely to fail
        print("Initializing Supabase client...")
        try:
            supabase_client = get_supabase_client()
            print("Supabase client initialized successfully")
        except Exception as db_error:
            print(f"Error initializing Supabase client: {str(db_error)}")
            # Continue without raising to allow server to start even without DB
            supabase_client = None
            
        # Try to initialize the crawler, but make it optional
        print("Initializing crawler...")
        try:
            # Initialize the crawler
            crawler = AsyncWebCrawler(config=browser_config)
            await crawler.__aenter__()
            print("Crawler initialized successfully")
        except Exception as crawler_error:
            print(f"Error initializing crawler: {str(crawler_error)}")
            # Set crawler to None, MCP will still work but without crawler functionality
            crawler = None
        
        # Create context with current time as last health check
        context = Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            last_health_check=time.time()
        )
        
        # Only start health check if crawler was initialized
        if crawler:
            try:
                # Start background health check task
                asyncio.create_task(browser_health_check_loop(context))
                print("Browser health check started")
            except Exception as task_error:
                print(f"Failed to start health check: {str(task_error)}")
                # Continue without health check
        
        # Return the context - even with partial initialization, server will start
        print("MCP server context initialized, starting server...")
        yield context
    except Exception as e:
        print(f"Critical error during initialization: {str(e)}")
        if crawler:
            try:
                await crawler.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"Error during crawler cleanup: {str(cleanup_error)}")
    finally:
        # Clean up the crawler
        if crawler:
            try:
                await crawler.__aexit__(None, None, None)
                print("Crawler cleaned up successfully")
            except Exception as e:
                print(f"Error during crawler cleanup: {str(e)}")

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')


def retry_async(max_retries=3, initial_delay=1, backoff_factor=2, exception_types=(Exception,)):
    """
    A decorator for retrying asynchronous functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which the delay increases each retry
        exception_types: Tuple of exception types to catch and retry
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    return await func(*args, **kwargs)
                except exception_types as e:
                    last_exception = e
                    # Don't sleep if this was the last attempt
                    if attempt < max_retries:
                        # Check if it's a Playwright 'target closed' error
                        is_target_closed = isinstance(e, PlaywrightError) and "Target page, context or browser has been closed" in str(e)
                        
                        # Log the retry attempt
                        error_type = "browser disconnect" if is_target_closed else type(e).__name__
                        print(f"Retry {attempt+1}/{max_retries} after {error_type} error: {str(e)}")
                        
                        # Sleep with exponential backoff
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
            
            # If we get here, all retries failed
            print(f"All {max_retries} retries failed. Last error: {str(last_exception)}")
            raise last_exception
        
        return wrapper
    return decorator

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

async def browser_health_check_loop(context: Crawl4AIContext):
    """
    Background task that periodically checks browser health.
    If the browser is disconnected, it will attempt to recreate it.
    
    Args:
        context: The Crawl4AI context with crawler instance
    """
    while True:
        try:
            # Wait for the health check interval
            await asyncio.sleep(context.health_check_interval)
            
            # Check if it's time for a health check
            current_time = time.time()
            if current_time - context.last_health_check >= context.health_check_interval:
                print("Performing browser health check...")
                
                # Test the browser connection with a simple operation
                try:
                    # Attempt to create a new page to verify browser is still working
                    browser = context.crawler.browser
                    if browser:
                        page = await browser.new_page()
                        await page.goto("about:blank", timeout=10000)
                        await page.close()
                        print("Browser health check successful")
                    else:
                        print("Browser reference is None, attempting to recreate crawler")
                        raise Exception("Browser reference is None")
                    
                except Exception as e:
                    print(f"Browser health check failed: {str(e)}")
                    print("Attempting to recreate browser...")
                    
                    # Try to close the existing browser
                    try:
                        if context.crawler:
                            await context.crawler.__aexit__(None, None, None)
                    except Exception as close_error:
                        print(f"Error closing existing browser: {str(close_error)}")
                    
                    # Create a new browser
                    try:
                        browser_config = BrowserConfig(
                            headless=True,
                            verbose=False
                        )
                        
                        new_crawler = AsyncWebCrawler(config=browser_config)
                        await new_crawler.__aenter__()
                        
                        # Replace the crawler in the context
                        context.crawler = new_crawler
                        print("Browser successfully recreated")
                    except Exception as recreate_error:
                        print(f"Failed to recreate browser: {str(recreate_error)}")
                
                # Update the last health check time
                context.last_health_check = current_time
        
        except asyncio.CancelledError:
            print("Browser health check task cancelled")
            break
        except Exception as e:
            print(f"Error in browser health check loop: {str(e)}")
            # Wait a bit before trying again
            await asyncio.sleep(60)

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    # Get components from context
    crawler = ctx.request_context.lifespan_context.crawler
    supabase_client = ctx.request_context.lifespan_context.supabase_client
    
    # Check if crawler is available
    if not crawler:
        return json.dumps({
            "success": False,
            "url": url,
            "error": "Crawler is not available. The server started without a crawler due to initialization issues."
        }, indent=2)
    
    # Check if supabase client is available
    if not supabase_client:
        return json.dumps({
            "success": False,
            "url": url,
            "error": "Supabase client is not available. The server started without database connection due to initialization issues."
        }, indent=2)
    
    # Track retry attempts
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # Configure the crawl
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS, 
                stream=False
            )
            
            # Crawl the page
            result = await crawler.arun(url=url, config=run_config)
            
            if result.success and result.markdown:
                # Chunk the content
                chunks = smart_chunk_markdown(result.markdown)
                
                # Prepare data for Supabase
                urls = []
                chunk_numbers = []
                contents = []
                metadatas = []
                
                for i, chunk in enumerate(chunks):
                    urls.append(url)
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    
                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = url
                    meta["source"] = urlparse(url).netloc
                    meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                    metadatas.append(meta)
                
                # Create url_to_full_document mapping
                url_to_full_document = {url: result.markdown}
                
                # Add to Supabase
                try:
                    # Use a smaller batch size for more reliable processing
                    add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=10)
                    
                    return json.dumps({
                        "success": True,
                        "url": url,
                        "chunks_stored": len(chunks),
                        "content_length": len(result.markdown),
                        "links_count": {
                            "internal": len(result.links.get("internal", [])),
                            "external": len(result.links.get("external", []))
                        },
                        "retry_count": retry_count
                    }, indent=2)
                except Exception as supabase_error:
                    return json.dumps({
                        "success": False,
                        "url": url,
                        "error": f"Database error: {str(supabase_error)}"
                    }, indent=2)
            else:
                # Check if we should retry
                retry_count += 1
                last_error = result.error_message
                
                if retry_count <= max_retries:
                    print(f"Retry {retry_count}/{max_retries} for URL {url} due to crawl failure: {last_error}")
                    await asyncio.sleep(2 * retry_count)  # Exponential backoff
                    continue
                else:
                    return json.dumps({
                        "success": False,
                        "url": url,
                        "error": last_error,
                        "retries_exhausted": True
                    }, indent=2)
        except PlaywrightError as e:
            # Specifically handle Playwright errors
            retry_count += 1
            last_error = str(e)
            
            if "Target page, context or browser has been closed" in last_error and retry_count <= max_retries:
                print(f"Retry {retry_count}/{max_retries} for URL {url} after browser disconnect: {last_error}")
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
                continue
            elif retry_count <= max_retries:
                print(f"Retry {retry_count}/{max_retries} for URL {url} after Playwright error: {last_error}")
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
                continue
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Playwright error after {max_retries} retries: {last_error}"
                }, indent=2)
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            
            if retry_count <= max_retries:
                print(f"Retry {retry_count}/{max_retries} for URL {url} after error: {last_error}")
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
                continue
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Error after {max_retries} retries: {last_error}"
                }, indent=2)
    
    # This should not be reached, but just in case
    return json.dumps({
        "success": False,
        "url": url,
        "error": f"Unknown error after {max_retries} retries"
    }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 5, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default reduced to 5)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)
    
    Returns:
        JSON string with crawl summary and storage information
    """
    # Get components from context
    crawler = ctx.request_context.lifespan_context.crawler
    supabase_client = ctx.request_context.lifespan_context.supabase_client
    
    # Check if crawler is available
    if not crawler:
        return json.dumps({
            "success": False,
            "url": url,
            "error": "Crawler is not available. The server started without a crawler due to initialization issues."
        }, indent=2)
    
    # Check if supabase client is available
    if not supabase_client:
        return json.dumps({
            "success": False,
            "url": url,
            "error": "Supabase client is not available. The server started without database connection due to initialization issues."
        }, indent=2)
    
    # Track retry attempts
    max_retries = 2  # For the overall operation
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            crawl_results = []
            crawl_type = "webpage"
            
            # Detect URL type and use appropriate crawl method
            if is_txt(url):
                # For text files, use simple crawl
                crawl_results = await crawl_markdown_file(crawler, url)
                crawl_type = "text_file"
            elif is_sitemap(url):
                # For sitemaps, extract URLs and crawl in parallel
                try:
                    sitemap_urls = parse_sitemap(url)
                    if not sitemap_urls:
                        return json.dumps({
                            "success": False,
                            "url": url,
                            "error": "No URLs found in sitemap"
                        }, indent=2)
                    
                    # Limit the number of URLs to crawl to prevent resource exhaustion
                    max_urls_to_crawl = 100  # Reasonable limit to prevent overwhelming the system
                    if len(sitemap_urls) > max_urls_to_crawl:
                        print(f"Limiting sitemap crawl from {len(sitemap_urls)} to {max_urls_to_crawl} URLs")
                        sitemap_urls = sitemap_urls[:max_urls_to_crawl]
                    
                    crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
                    crawl_type = "sitemap"
                except Exception as sitemap_error:
                    print(f"Error processing sitemap: {str(sitemap_error)}")
                    if retry_count < max_retries:
                        retry_count += 1
                        await asyncio.sleep(2 * retry_count)  # Exponential backoff
                        continue
                    else:
                        return json.dumps({
                            "success": False,
                            "url": url,
                            "error": f"Sitemap processing error: {str(sitemap_error)}"
                        }, indent=2)
            else:
                # For regular URLs, use recursive crawl
                crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
                crawl_type = "webpage"
            
            if not crawl_results:
                # Try again if we didn't get any results
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"Retry {retry_count}/{max_retries} for URL {url} - no content found")
                    await asyncio.sleep(2 * retry_count)  # Exponential backoff
                    continue
                else:
                    return json.dumps({
                        "success": False,
                        "url": url,
                        "error": "No content found after multiple attempts"
                    }, indent=2)
            
            # Process results and store in Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            chunk_count = 0
            
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
                
                for i, chunk in enumerate(chunks):
                    urls.append(source_url)
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    
                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = source_url
                    meta["source"] = urlparse(source_url).netloc
                    meta["crawl_type"] = crawl_type
                    meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                    metadatas.append(meta)
                    
                    chunk_count += 1
            
            # Create url_to_full_document mapping
            url_to_full_document = {}
            for doc in crawl_results:
                url_to_full_document[doc['url']] = doc['markdown']
            
            # Add to Supabase in smaller batches
            batch_size = 10  # Smaller batch size for more reliability
            
            try:
                add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=batch_size)
                
                return json.dumps({
                    "success": True,
                    "url": url,
                    "crawl_type": crawl_type,
                    "pages_crawled": len(crawl_results),
                    "chunks_stored": chunk_count,
                    "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else []),
                    "retry_count": retry_count
                }, indent=2)
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")
                # If this is a database error, return failure immediately
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Database error: {str(db_error)}"
                }, indent=2)
                
        except PlaywrightError as e:
            retry_count += 1
            last_error = str(e)
            
            if retry_count <= max_retries:
                print(f"Retry {retry_count}/{max_retries} for URL {url} after Playwright error: {last_error}")
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
                continue
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Playwright error after {max_retries} retries: {last_error}"
                }, indent=2)
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            
            if retry_count <= max_retries:
                print(f"Retry {retry_count}/{max_retries} for URL {url} after error: {last_error}")
                await asyncio.sleep(2 * retry_count)  # Exponential backoff
                continue
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Error after {max_retries} retries: {last_error}"
                }, indent=2)
    
    # This should not be reached, but just in case
    return json.dumps({
        "success": False,
        "url": url,
        "error": f"Unknown error after {max_retries} retries"
    }, indent=2)

@retry_async(max_retries=3, initial_delay=2, exception_types=(PlaywrightError, asyncio.TimeoutError))
async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file with retry logic.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    # Use a more robust configuration
    crawl_config = CrawlerRunConfig()

    try:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{'url': url, 'markdown': result.markdown}]
        else:
            print(f"Failed to crawl {url}: {result.error_message}")
            return []
    except Exception as e:
        print(f"Exception during markdown file crawl for {url}: {str(e)}")
        raise  # Re-raise to trigger retry if needed

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions (default reduced to 5)
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    # Use more conservative settings for the crawler config
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, 
        stream=False
    )
    
    # Use more conservative memory management settings
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=60.0,  # Lower threshold for memory usage
        check_interval=0.5,  # Check more frequently
        max_session_permit=max_concurrent
    )

    try:
        results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
        return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
    except Exception as e:
        print(f"Error during batch crawl: {str(e)}")
        # Return partial results if available
        if isinstance(results, list) and results:
            return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
        return []

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions (default reduced to 5)
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    # Use conservative settings for the crawler config
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, 
        stream=False
    )
    
    # Use more conservative memory management settings
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=60.0,  # Lower threshold for memory usage
        check_interval=0.5,  # Check more frequently
        max_session_permit=max_concurrent
    )

    visited = set()
    error_urls = set()  # Track URLs that failed to prevent retrying problematic URLs

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        # Filter out URLs that have already been visited or caused errors
        urls_to_crawl = [normalize_url(url) for url in current_urls 
                         if normalize_url(url) not in visited and normalize_url(url) not in error_urls]
        
        if not urls_to_crawl:
            break

        try:
            # Break URLs into smaller batches to reduce resource contention
            batch_size = min(len(urls_to_crawl), 10)  # Process at most 10 URLs at a time
            for i in range(0, len(urls_to_crawl), batch_size):
                batch_urls = urls_to_crawl[i:i+batch_size]
                
                # Process this batch
                results = await crawler.arun_many(urls=batch_urls, config=run_config, dispatcher=dispatcher)
                next_level_urls = set()

                for result in results:
                    norm_url = normalize_url(result.url)
                    visited.add(norm_url)

                    if result.success and result.markdown:
                        results_all.append({'url': result.url, 'markdown': result.markdown})
                        # Process internal links only if the page was successfully crawled
                        for link in result.links.get("internal", []):
                            try:
                                next_url = normalize_url(link["href"])
                                if next_url not in visited and next_url not in error_urls:
                                    next_level_urls.add(next_url)
                            except Exception as e:
                                print(f"Error processing link {link}: {str(e)}")
                    else:
                        # Mark failed URLs to avoid retry
                        error_urls.add(norm_url)
                        print(f"Failed to crawl {result.url}: {result.error_message}")

                current_urls = next_level_urls
                
                # Add a small delay between batches to allow resources to be freed
                if i + batch_size < len(urls_to_crawl):
                    await asyncio.sleep(1)
                    
        except Exception as e:
            print(f"Error during recursive crawl at depth {depth}: {str(e)}")
            # Continue with the next depth level despite errors

    return results_all

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources
    """
    # Get the Supabase client from the context
    supabase_client = ctx.request_context.lifespan_context.supabase_client
    
    # Check if supabase client is available
    if not supabase_client:
        return json.dumps({
            "success": False,
            "error": "Supabase client is not available. The server started without database connection due to initialization issues."
        }, indent=2)
    
    try:
        # Use a direct query with the Supabase client
        # This could be more efficient with a direct Postgres query but
        # I don't want to require users to set a DB_URL environment variable as well
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()
            
        # Use a set to efficiently track unique sources
        unique_sources = set()
        
        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)
        
        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    # Get the Supabase client from the context
    supabase_client = ctx.request_context.lifespan_context.supabase_client
    
    # Check if supabase client is available
    if not supabase_client:
        return json.dumps({
            "success": False,
            "query": query,
            "error": "Supabase client is not available. The server started without database connection due to initialization issues."
        }, indent=2)
    
    try:
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        # Perform the search
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

class SafeSSETransport:
    """
    A wrapper around the SSE transport to handle BrokenResourceError more gracefully.
    This helps prevent the ASGI application from crashing when browser sessions are closed.
    """
    def __init__(self, mcp_server: FastMCP):
        self.mcp_server = mcp_server
        self.max_consecutive_failures = 10
        self.restart_delay = 2
        self.consecutive_failures = 0
    
    async def run(self):
        """
        Run the MCP server with enhanced error handling for SSE transport.
        Catches BrokenResourceError and provides more graceful shutdown.
        """
        import traceback
        from anyio import BrokenResourceError
        
        try:
            # Reset failure counter on successful start
            self.consecutive_failures = 0
            await self.mcp_server.run_sse_async()
        except (BrokenResourceError, RuntimeError) as e:
            # Include RuntimeError which is happening during initialization
            self.consecutive_failures += 1
            print(f"Caught error in SSE transport ({self.consecutive_failures}/{self.max_consecutive_failures}): {str(e)}")
            
            if "Received request before initialization was complete" in str(e):
                print("This is likely due to initialization issues. Retrying after a delay.")
            elif isinstance(e, BrokenResourceError):
                print("This is likely due to a browser session being closed unexpectedly.")
                
            print(f"The server will wait {self.restart_delay} seconds and then continue running.")
            
            # Retry with exponential backoff if we haven't exceeded max failures
            if self.consecutive_failures < self.max_consecutive_failures:
                await asyncio.sleep(self.restart_delay)
                self.restart_delay = min(self.restart_delay * 1.5, 30)  # Increase delay up to 30 seconds max
                await self.run()  # Recursive call to restart
            else:
                print(f"Exceeded maximum consecutive failures ({self.max_consecutive_failures}). Server will not auto-restart.")
                print("Please check your configuration and restart the server manually.")
        except asyncio.CancelledError:
            # Gracefully handle cancellation
            print("Server shutdown requested.")
            raise
        except Exception as e:
            self.consecutive_failures += 1
            print(f"Unhandled exception in SSE transport ({self.consecutive_failures}/{self.max_consecutive_failures}): {str(e)}")
            traceback.print_exc()
            
            # Retry with exponential backoff if we haven't exceeded max failures
            if self.consecutive_failures < self.max_consecutive_failures:
                print(f"The server will wait {self.restart_delay} seconds and then continue running.")
                await asyncio.sleep(self.restart_delay)
                self.restart_delay = min(self.restart_delay * 2, 60)  # Increase delay up to 60 seconds max
                await self.run()  # Recursive call to restart
            else:
                print(f"Exceeded maximum consecutive failures ({self.max_consecutive_failures}). Server will not auto-restart.")
                print("Please check logs for errors and restart the server manually.")

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    print(f"Starting MCP server with {transport} transport...")
    
    if transport == 'sse':
        # Run the MCP server with the safer SSE transport wrapper
        try:
            safe_transport = SafeSSETransport(mcp)
            print("SSE transport initialized, starting server...")
            await safe_transport.run()
        except Exception as e:
            print(f"Critical error in main SSE transport: {str(e)}")
            import traceback
            traceback.print_exc()
            # Sleep briefly to allow logs to be written before exiting
            await asyncio.sleep(2)
    else:
        # Run the MCP server with stdio transport
        try:
            print("Starting stdio transport...")
            await mcp.run_stdio_async()
        except Exception as e:
            print(f"Critical error in stdio transport: {str(e)}")
            import traceback
            traceback.print_exc()
            # Sleep briefly to allow logs to be written before exiting
            await asyncio.sleep(2)

if __name__ == "__main__":
    print(f"Starting Crawl4AI RAG MCP Server on port {os.getenv('PORT', '8051')}...")
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error starting server: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Server shutdown complete.")
        # Force exit to avoid hanging threads
        os._exit(0)