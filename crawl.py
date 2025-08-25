import asyncio

from crawlee import Glob
from crawlee.crawlers import BeautifulSoupCrawler, BeautifulSoupCrawlingContext


async def main() -> None:
    crawler = BeautifulSoupCrawler(
        max_requests_per_crawl=50,
    )

    @crawler.router.default_handler
    async def request_handler(context: BeautifulSoupCrawlingContext) -> None:
        context.log.info(f'Processing {context.request.url} ...')
        context.log.info(f"Queueing from {context.request.url}")
        for a_tag in context.soup.find_all('a', href=True):
            context.log.debug(f"Found href: {a_tag['href']}")

        # Check if this page links to /download
        for a_tag in context.soup.find_all('a', href=True):
            href = a_tag['href']
            context.log.debug(f'Found link: {href}')
            if '/download' in href:
                context.log.info(f'{context.request.url} links to {href}')

        links = context.soup.find_all('a', href=True)
        for a in links:
            context.log.debug(f"Raw href: {a['href']}")
        # Only follow internal links (same domain)
        await context.enqueue_links(
            strategy='same-domain',
            base_url=context.request.url,
        )

    await crawler.run(['https://scylladb.com'])


if __name__ == '__main__':
    asyncio.run(main())