from selenium.webdriver.chrome.options import Options

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import glob

wordList=[
    "Outside", "Ancient", "Parents", "Annual", "Request", "Another", "Roundabout", "Any", "Say", "Because", "She",
    "Both", "Sister", "Brain", "So (Accentuator)", "Children", "So (In Order To)", "Come", "Some",
    "Continuously", "Soon", "Do", "Subsequent", "Dry", "Subway", "Elevator", "Sufficient", "Empty", "There",
    "Eye", "This", "Father", "Thoughtful", "Few", "Tongue", "From", "Trust", "Go", "Truthful", "He",
    "Universal", "Hear", "Up", "Heart", "Upward", "However", "Usually", "I", "Walk", "Jeep", "Warm", "Knock",
    "We", "Lakh", "Weak", "Literally", "Without", "Mehr", "Woman", "Mine", "Work", "Mother", "Worthy",
    "Mouth", "Yellow Light", "Move", "You"
]


def wait_for_download(download_dir, timeout=30):
    initial_files = set(glob.glob(os.path.join(download_dir, "*")))
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_files = set(glob.glob(os.path.join(download_dir, "*")))
        new_files = current_files - initial_files
        if new_files:
            print(f"✅ Download completed: {new_files}")
            return True
        time.sleep(1)
    print("❌ Download timeout.")
    return False

path = r"C:\Users\Dell\Documents\chromedriver-win64\chromedriver-win64\chromedriver.exe"
download_dir = r"C:\Users\Dell\Documents\Github Repos\Sign-Language-Recognition-Vid-to-Image-\data\PSL_Dictionary"

chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True
})

service = Service(path)
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.get("https://psl.org.pk")
Psl_Dictionary=WebDriverWait(driver,25).until(EC.presence_of_element_located((By.XPATH,"//a[@href='/dictionary']")))
driver.execute_script("arguments[0].click();", Psl_Dictionary)
driver.implicitly_wait(10)




error_log_path = os.path.join(download_dir, "failed_downloads.txt")

# Get already downloaded words (by filename, ignoring extension)
downloaded_words = set()
for f in os.listdir(download_dir):
    name, ext = os.path.splitext(f)
    downloaded_words.add(name.lower())

for word in wordList:
    safe_word = str(word).replace("/", "_").replace("\\", "_").replace(" ", "_")  # Make filename safe
    if safe_word.lower() in downloaded_words:
        print(f"⏩ Skipping already downloaded: {word}")
        continue
    try:
        search_toggle=WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "search-toggle")))
        search_toggle.click()

        search_bar = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='collapseExample']/div/input")))
        search_bar.clear()
        search_bar.send_keys(word)

        results = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "possible-search-items"))
        )
        print([item.text for item in results])
        target_text = str(word).lower()
        viewport = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "cdk-virtual-scroll-viewport"))
        )

        scroll_step = 30
        max_scroll_attempts = 100
        found = False

        for attempt in range(max_scroll_attempts):
            items = driver.find_elements(By.CLASS_NAME, "possible-search-items")
            match_index = None
            for idx, item in enumerate(items):
                visible_text = item.text.strip().split(")", 1)[-1].strip().lower()
                if visible_text == target_text:
                    match_index = idx
                    item_text = item.text.strip()
                    break
            if match_index is not None:
                items = driver.find_elements(By.CLASS_NAME, "possible-search-items")
                item = items[match_index]
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", item)
                print(f"✅ Clicked on: {item_text}")
                found = True
                break
            driver.execute_script("arguments[0].scrollTop += arguments[1];", viewport, scroll_step)
            time.sleep(0.3)
        if not found:
            print(f"❌ '{word}' not found after {max_scroll_attempts} scrolls.")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"Not found: {word}\n")
            continue
        driver.implicitly_wait(10)
        try:
            download_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/app-root/div[2]/app-player/div[4]/div[3]/div[1]/div[2]/div[2]/div/div/a[2]/i")))
            driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", download_button)

            quality_button = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "/html/body/app-root/div[2]/app-player/div[2]/div/div/form/div[2]/div[1]/div/label")))
            driver.execute_script("arguments[0].scrollIntoView(true);", quality_button)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", quality_button)

            download_button2= WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//*[@id='exampleModal-vds']/div/div/form/div[3]/button")))
            driver.execute_script("arguments[0].scrollIntoView(true);", download_button2)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", download_button2)

            # Wait for download and rename file
            before_files = set(os.listdir(download_dir))
            if not wait_for_download(download_dir):
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"Download timeout: {word}\n")
            else:
                after_files = set(os.listdir(download_dir))
                new_files = after_files - before_files
                if new_files:
                    # Rename the most recent file to the word
                    newest_file = max([os.path.join(download_dir, f) for f in new_files], key=os.path.getctime)
                    ext = os.path.splitext(newest_file)[1]
                    new_name = os.path.join(download_dir, f"{safe_word}{ext}")
                    try:
                        os.rename(newest_file, new_name)
                        print(f"📝 Renamed {newest_file} to {new_name}")
                    except Exception as e:
                        print(f"❌ Rename error for '{word}': {e}")
                        with open(error_log_path, "a", encoding="utf-8") as f:
                            f.write(f"Rename error: {word} - {e}\n")
        except Exception as e:
            print(f"❌ Download error for '{word}': {e}")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"Download error: {word} - {e}\n")
        driver.implicitly_wait(300)
    except Exception as e:
        print(f"❌ Error for '{word}': {e}")
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"General error: {word} - {e}\n")

  # Wait for the download to start

