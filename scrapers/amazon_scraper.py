import amazonproduct
from pandas import DataFrame
from lxml import html
import requests

# TODO: Get link to buy. item.ItemLinks.ItemLink.URL


def get_reviews(url):
	"""
	Mini scraper to get reviews from url
	"""
	page = requests.get(url)
	tree = html.fromstring(page.content)
	review_list = tree.xpath('//div[@class="reviewText"]/text()')
	return review_list


class AmazonScraper:
	"""
	Amazon class, initializes API.
	"""
	def __init__(self):
		config = {}
		execfile("config.py", config)
		amazonproduct.HOSTS['mx'] = 'webservices.amazon.com.mx'
		self.api = amazonproduct.API(cfg={'access_key': config['AWS_ACCESS_KEY'],
										'secret_key': config['AWS_SECRET_KEY'], 
										'associate_tag': config['AWS_ASSOCIATE_TAG'],
										'locale': 'mx'})

	def scrape_items(self, category, search):
		"""
		Scrapes items using API.
		:param category: Item category e.g. books, electronics: str
		:param search: title of item to search: str
		"""
		items = self.api.item_search(search_index=category, ResponseGroup='Large', Title=search)
		df = DataFrame()
		for item in items:
			results = {}
			results["title"] = unicode(item.ItemAttributes.Title) if hasattr(item.ItemAttributes, 'Title') else None
			results["price"] = unicode(item.ItemAttributes.ListPrice.FormattedPrice) if hasattr(item.ItemAttributes, 'ListPrice') else None
			results["image_url"] = item.LargeImage.URL.text if hasattr(item, 'LargeImage') else None
			results["reviews_url"] = item.CustomerReviews.IFrameURL.text if item.CustomerReviews.HasReviews else None
			results['reviews'] = get_reviews(results["reviews_url"]) if results["reviews_url"] else None
			df=df.append(other=results, ignore_index=True)
		return df

def main():
	# initialize amazon class
	amz = AmazonScraper()
	# scrape items
	search = 'Feynman Lectures'
	category = "Books"
	df = amz.scrape_items(category=category, search=search)


if __name__ == "__main__":
    main()
