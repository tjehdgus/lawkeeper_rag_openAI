import os
import time
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class QALawCrawler:
    def __init__(self, start_url="https://www.klac.or.kr/legalinfo/counsel.do"):
        self.start_url = start_url
        self.driver = None

    def give_options(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--single-process")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        return options

    def start_driver(self):
        self.driver = webdriver.Chrome(options=self.give_options())

    def quit_driver(self):
        if self.driver:
            self.driver.quit()

    def crawlling_data(self):
        self.driver.get(self.start_url)
        time.sleep(1)  # 페이지 초기 로딩 대기

        data_list = []
        end_point = False
        page_count = 1
        
        while not end_point:
            for j in range(2, 11):  # 페이지 내 숫자 링크 (2~10)
                print(f"\n📄 현재 페이지 세트: {page_count}")
                for k in range(1, 11):  # 각 페이지의 게시글 (1~10)

                    try:
                        self._wait_driver_click(
                            f'//*[@id="content"]/div[2]/div/form/div[2]/table/tbody/tr[{k}]/td[2]/a'
                        )
                        data = self._collect_data()
                        if data:
                            data_list.append(data)
                        self._wait_driver_click('//*[@id="content"]/div[2]/div/div/a')
                    except Exception as e:
                        print(f"[!] 내부 페이지 접근 실패: {e}")
                        break

                try:
                    self._wait_driver_click(f'//*[@id="content"]/div[2]/div/form/div[3]/a[{j}]')
                    page_count += 1
                except:
                    end_point = True
                    break

            try:
                self._wait_driver_click('//*[@id="content"]/div[2]/div/form/div[3]/button[3]')
            except:
                break

        self._save_data(data_list=data_list)

    def _wait_driver_click(self, xpath):
        WebDriverWait(self.driver, timeout=10).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        ).click()
        time.sleep(0.5)  # 클릭 후 로딩 대기

    def _collect_data(self):
        try:
            html = self.driver.page_source
            soup = bs(html, "html.parser")
            data = []
            for i in range(1, 5):
                dd = soup.select_one(f"#print_page > div:nth-child({i}) > dl > dd")
                if dd:
                    data.append(dd.text.strip())
                else:
                    data.append("")
            data.append(self.driver.current_url)
            return data
        except Exception as e:
            print(f"[!] 데이터 수집 실패: {e}")
            return ["", "", "", "", self.driver.current_url]

    def _save_data(self, data_list):
        df = pd.DataFrame(
            data=data_list,
            columns=["division", "title", "question", "answer", "url"]
        )
        save_path = "d:\\법률\\law_qa.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[✓] 저장 완료: {save_path}")


if __name__ == "__main__":
    crawler = QALawCrawler()
    crawler.start_driver()
    try:
        crawler.crawlling_data()
    finally:
        crawler.quit_driver()