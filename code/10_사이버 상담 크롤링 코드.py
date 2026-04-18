import logging
import os
import pickle
import re
import time
from contextlib import contextmanager
from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.remote_connection import LOGGER as selenium_logger
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from tqdm.auto import tqdm
from webdriver_manager.chrome import ChromeDriverManager
from utils import utilize_loggers

selenium_logger.setLevel(logging.WARNING)
os.environ["WDM_LOG"] = "0"

@contextmanager
def timer():
    t0 = time.time()
    yield lambda: time.time() - t0

def measure_elapsed_time(timer_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with timer() as elapsed_time:
                result = func(*args, **kwargs)
                logger.info(f"{elapsed_time():>8.3f} seconds elapsed @ {timer_name}")
            return result
        return wrapper
    return decorator

class QADataCrawler:
    def __init__(self,
        board_url="https://www.klac.or.kr/legalstruct/cyberConsultation/selectOpenArticleList.do?boardCode=3",
        base_url="https://www.klac.or.kr/legalstruct/cyberConsultation/selectOpenArticleDetail.do?boardCode=3&contentId=",
    ):
        self.driver = None
        self.board_url = board_url
        self.base_url = base_url

    def start_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--start-maximized")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        logger.info("✅ Chrome Driver Started")

    def quit_driver(self):
        self.driver.quit()
        logger.info("🛑 크롬 드라이버 종료됨")

    @measure_elapsed_time("Total Crawling Process")
    def get_data(self):
        logger.info("🚀 크롤링 시작")
        case_ids = self._get_all_case_ids()
        logger.info(f"🔍 수집된 케이스 ID 수: {len(case_ids)}")
        case_info = self._get_all_case_contents(case_ids)
        self._save_dataframe(case_info)

    def _get_case_id(self):
        case_ids = []
        WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@onclick, 'fn_inquire_detail')]")))
        elements = self.driver.find_elements(By.XPATH, "//a[contains(@onclick, 'fn_inquire_detail')]")
        for e in elements:
            onclick = e.get_attribute("onclick")
            match = re.search(r"fn_inquire_detail\('\d+',\s*'(US_\d+)'\)", onclick)
            if match:
                case_ids.append(match.group(1))
        return case_ids

    @measure_elapsed_time("Get all case ids")
    def _get_all_case_ids(self, save_id_list=False):
        self.driver.get(self.board_url)
        time.sleep(3)
        case_ids = []
        visited_onclick = set()
        page_counter = 0

        while page_counter < 300:
            logger.info(f"📄 페이지 {page_counter + 1} 처리 중...")
            try:
                new_ids = self._get_case_id()
                case_ids.extend(new_ids)
                logger.info(f"  ✅ 현재 누적 ID 수: {len(case_ids)}")
            except Exception as e:
                logger.warning(f"⚠️ ID 수집 실패: {e}")
                break

            try:
                current_page = self.driver.find_element(By.CSS_SELECTOR, ".paging_wrap a.current")
                current_num = int(current_page.text.strip())

                found = False
                page_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".paging_wrap a")
                for btn in page_buttons:
                    if btn.text.strip().isdigit() and int(btn.text.strip()) == current_num + 1:
                        self.driver.execute_script("arguments[0].click();", btn)
                        time.sleep(2.5)
                        found = True
                        break

                if not found:
                    try:
                        next_btn = self.driver.find_element(By.CSS_SELECTOR, "button.btn_page_next")
                        self.driver.execute_script("arguments[0].click();", next_btn)
                        time.sleep(2.5)
                    except:
                        logger.info("ℹ️ 더 이상 이동할 페이지 없음")
                        break
            except Exception as e:
                logger.warning(f"⚠️ 페이지 이동 중 오류 발생: {e}")
                break

            page_counter += 1

        if save_id_list:
            self._save_case_id_list(case_ids, "case_id_list.pkl")

        return list(dict.fromkeys(case_ids))

    def _get_case_content_by_id(self, case_id):
        url = self.base_url + case_id
        try:
            self.driver.execute_script("window.open(arguments[0]);", url)
            self.driver.switch_to.window(self.driver.window_handles[-1])
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "view_head")))

            case_title = self.driver.find_element(By.CLASS_NAME, "view_head").text.strip()
            date_created = self.driver.find_element(By.XPATH, "//dt[text()='신청일']/following-sibling::dd").text.strip()
            date_answered = self.driver.find_element(By.XPATH, "//dt[text()='답변일자']/following-sibling::dd").text.strip()
            content_blocks = self.driver.find_elements(By.CLASS_NAME, "notice_contents")
            content_text = content_blocks[0].text.strip() if len(content_blocks) > 0 else ""
            answer_text = content_blocks[1].text.strip() if len(content_blocks) > 1 else ""

            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])

            return [case_title, date_created, date_answered, content_text, answer_text]
        except Exception as e:
            logger.warning(f"❌ 케이스 {case_id} 처리 실패: {e}")
            try:
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
            except:
                pass
            return None

    @measure_elapsed_time("Get all case contents")
    def _get_all_case_contents(self, case_ids):
        case_info = []
        for idx, case_id in enumerate(tqdm(case_ids)):
            logger.info(f"🔎 ({idx+1}/{len(case_ids)}) 케이스 ID: {case_id} 처리 중...")
            info = self._get_case_content_by_id(case_id)
            if info:
                case_info.append(info)
        return case_info

    def _save_dataframe(self, case_info):
        df = pd.DataFrame(case_info, columns=["case_title", "date_created", "date_answered", "content", "answer"])
        save_path = r"d:\\법률\\raw_qa_dataset.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.sort_values(by="date_created", inplace=True)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 저장 완료: {save_path} (총 {len(df)}건)")

    def _save_case_id_list(self, case_id_list, file_name="case_id_list.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(case_id_list, f)

    def _load_case_id_list(self, file="case_id_list.pkl"):
        with open(file, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    logger = utilize_loggers("jupyter_run")
    crawler = QADataCrawler()
    try:
        crawler.start_driver()
        crawler.get_data()
    finally:
        crawler.quit_driver()