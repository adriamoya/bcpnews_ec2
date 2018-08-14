import sys
import datetime

def get_date_input():

    try:
        # Input control: crawl date
        if len(sys.argv) > 1: # if passed as an argument
            if len(sys.argv[1]) == 8: # valid YYYYMMDD date format
                crawl_date_input = sys.argv[1]
                crawl_date = datetime.datetime.strptime(crawl_date_input,"%Y%m%d")
                print("\nCrawling datetime is:", crawl_date.strptime(crawl_date_input,"%Y%m%d"), "\n")
            else:
                raise ValueError('The input date format expected is YYYYMMDD. Please try again.')

        else: # if no argument specified
            crawl_date = datetime.datetime.today()
            print("\nCrawling datetime not specified. Crawling newspapers for today:", crawl_date, "\n")
    except:
        crawl_date = datetime.datetime.today()
        print("\nCrawling datetime not specified. Crawling newspapers for today:", crawl_date, "\n")

    return crawl_date
