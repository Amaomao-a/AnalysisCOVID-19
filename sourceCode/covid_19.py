import unittest
import datetime
import time
import pandas as pd
from lxml import etree
from selenium import webdriver

'''
# 为了将Chrome不弹出界面，实现无界面爬取
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

self.browser = webdriver.Chrome(options=chrome_options)  # 创建谷歌浏览器对象
'''
class MySelenium(unittest.TestCase):
    def setUp(self):
        self.browser = webdriver.Chrome()
        self.date = datetime.datetime.now().strftime('%Y-%m-%d')  # 日期
        self.country = []  # 国家
        self.new_confirm = []  # 今日确诊
        self.total_confirm = []  # 累计确诊
        self.suspect = []  # 今日疑似
        self.dead = []  # 累计死亡
        self.heal = []  # 累计治愈

        # 请求网页
        self.browser.get('http://news.ifeng.com/c/special/7uLj4F83Cqm')
        # 模拟浏览器手动点击'查看更多'
        self.browser.find_element_by_xpath('//*[@id="list"]/div[2]/div[16]/span').click()

    def test(self):
        time.sleep(3)  # 让网页完全加载，避免网页没加载完导致获取的数据丢失
        html = self.browser.page_source  # 获取网页源码
        content = etree.HTML(html)  # 把源码解析为html dom文档

        # 使用xpath去匹配所有的信息
        country_list = content.xpath('//div[@class="tr_list_3X3bg3Ov"]/span[1]/text()')
        new_confirm_list = content.xpath('//div[@class="tr_list_3X3bg3Ov"]/span[2]/text()')
        total_confirm_list = content.xpath('//div[@class="tr_list_3X3bg3Ov"]/span[3]/text()')
        heal_list = content.xpath('//div[@class="tr_list_3X3bg3Ov"]/span[4]/text()')
        dead_list = content.xpath('//div[@class="tr_list_3X3bg3Ov"]/span[5]/text()')

        report = []
        # 对全球数据进行聚类划分生成数据行
        for i in range(len(country_list)):
            self.country.append(country_list[i])
            self.new_confirm.append(new_confirm_list[i])
            self.total_confirm.append(total_confirm_list[i])
            self.heal.append(heal_list[i])
            self.dead.append(dead_list[i])

            data = [self.date, self.country[i], self.new_confirm[i], self.total_confirm[i], self.heal[i], self.dead[i]]
            report.append(data)

        # 网页特点, 需要额外获取中国的数据
        self.browser.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/a/span').click()
        html = self.browser.page_source  # 获取网页源码
        content = etree.HTML(html)  # 把源码解析为html dom文档
        c = '中国'
        new_c = content.xpath('//div[@class="num4_zVWiof-O"]/span[1]/span[2]/text()')
        total_c = content.xpath('//div[@class="num4_zVWiof-O"]/span[2]/text()')
        h = content.xpath('//div[@class="num5_2Fsthu_Y"]/span[2]/text()')
        d = content.xpath('//div[@class="num6_lDtxv4aj"]/span[2]/text()')
        self.country.append(c)
        self.new_confirm.append(int(new_c[0]))
        self.total_confirm.append(total_c[0])
        self.heal.append(h[0])
        self.dead.append(d[0])
        data = [self.date, c, int(new_c[0]), total_c[0], h[0], d[0]]
        report.append(data)

        section_name = ['date', 'country', 'new_confirm', 'total_confirm', 'heal', 'dead']
        Data_csv = pd.DataFrame(columns=section_name, data=report)
        filename = 'Data' + str(self.date) + '.csv'
        Data_csv.to_csv(filename)

    def tearDown(self):
        self.browser.quit()
        print('Covid-19 data get successfully!')

'''
class getPopulation(unittest.TestCase):
    def setUp(self):
        self.browser = webdriver.Chrome()
        self.country = []  # 国家
        self.population = []  # 人口

        # 请求网页
        self.browser.get('https://www.phb123.com/city/renkou/rk.html')

    def testPopulation(self):
        count = 1
        rec = []
        while count <= 12:  # 获取网页列出的所有国家
            html = self.browser.page_source  # 获取网页源码
            content = etree.HTML(html)  # 把源码解析为html dom文档
            # 使用xpath去匹配所有的信息
            country_tmp = content.xpath('//span[@class="fl"]/../p/text()')
            popu_tmp = content.xpath('//span[@class="fl"]/../../../td[3]/text()')

            for i in range(len(popu_tmp)):
                data = [country_tmp[i], popu_tmp[i]]
                self.country.append(country_tmp[i])
                self.population.append(popu_tmp[i])
                rec.append(data)

            # 模拟浏览器手动点击'下一页'  tip:网页特殊，翻页可能会导致xpath改变，因此需要判断一下
            if count == 1 or count == 9:
                self.browser.find_element_by_xpath('//div[@class="page mt10"]/a[11]').click()
            elif 2 <= count <= 8:
                self.browser.find_element_by_xpath('//div[@class="page mt10"]/a[12]').click()
            elif count == 10:
                self.browser.find_element_by_xpath('//div[@class="page mt10"]/a[10]').click()
            elif count == 11:
                self.browser.find_element_by_xpath('//div[@class="page mt10"]/a[9]').click()
            count += 1

        section_name = ['country', 'population']
        Data_csv = pd.DataFrame(columns=section_name, data=rec)
        Data_csv.to_csv('countryMsg.csv')

    def tearDown(self):
        self.browser.quit()
        print('Population get successfully!')
  
'''  # 获取全球各国人口数量

unittest.main()