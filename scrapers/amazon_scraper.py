import amazonproduct
import pandas as pd

# TODO: Get price, title, image(s), link to buy and structure it. item.ItemLinks.ItemLink.URL
# TODO: Scrape review's urls :( 


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
		items = self.api.item_search(search_index=category, ResponseGroup='Large', Title=search)
		df = pd.DataFrame()
		for item in items:
			results = {}
			results["price"] = unicode(item.ItemAttributes.ListPrice.FormattedPrice) if hasattr(item.ItemAttributes, 'ListPrice') else None
			results["image"] = item.LargeImage.URL.text if hasattr(item, 'LargeImage') else None
			results["url"] = item.CustomerReviews.IFrameURL.text if item.CustomerReviews.HasReviews else None
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
