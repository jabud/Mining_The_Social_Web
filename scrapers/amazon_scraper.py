import amazonproduct

# TODO: Get price, title, image(s), link to buy and structure it.
# TODO: Paginate through at least 10 (maybe more)
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
items = api.item_search(search_index='Books', ResponseGroup='Reviews', Title="Quantum Theory Of Fields")
urls = []
for item in items:
    if item.CustomerReviews.HasReviews:
        urls.append(item.CustomerReviews.IFrameURL.text)
