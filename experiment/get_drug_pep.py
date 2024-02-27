
import os
import wget
import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
option = webdriver.ChromeOptions()
# I use the following options as my machine is a window subsystem linux.
# I recommend to use the headless option at least, out of the 3
# option.add_argument('--headless')
# option.add_argument('--no-sandbox')
# option.add_argument('--disable-dev-sh-usage')
option.add_argument('user-agent="Mozilla/5.0 (iPod; U; CPU iPhone OS 2_1 like Mac OS X; ja-jp) AppleWebKit/525.18.1 (KHTML, like Gecko) Version/3.1.1 Mobile/5F137 Safari/525.20"')
# option.add_argument('Referer="https://admetmesh.scbdd.com/service/evaluation/cal"')
# option.add_argument("--proxy-server=http://120.71.147.222:8901")

# Replace YOUR-PATH-TO-CHROMEDRIVER with your chromedriver location
# path=sys.args[0]




if __name__ == '__main__':
    import ssl
    import os

    os.environ['WDM_SSL_VERIFY'] = '0'
    # import urllib
    #
    ssl._create_default_https_context = ssl._create_unverified_context
    #
    ssl.match_hostname = lambda cert, hostname: True
    # url = 'https://go.drugbank.com/unearth/q?c=_score&d=down&query=drug+peptide&searcher=drugs'
    # html = urllib.request.urlopen(url).read().decode(encoding="ISO-8859-1")
    # # text = get_text(html)
    # # data = text.split()
    # print(html)
    with Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) as driver:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                   Object.defineProperty(navigator, 'webdriver', {
                     get: () => undefined
                   })
                 """
        })
        driver.get(
            'https://go.drugbank.com/unearth/q?c=_score&d=down&query=drug+peptide&searcher=drugs')  # Getting page HTML through request
        pageSource = driver.page_source
        pageSource.split("Caffeine")
        print(pageSource.split("Caffeine"))