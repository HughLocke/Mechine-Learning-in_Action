# -*-coding:utf-8 -*-
from bs4 import BeautifulSoup
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
   # 打开并读取HTML文件
   with open(inFile, encoding='utf-8') as f:
       html = f.read()
   soup = BeautifulSoup(html)
   i = 1
   # 根据HTML页面结构进行解析
   currentRow = soup.find_all('table', r = "%d" % i)
   while(len(currentRow) != 0):
       currentRow = soup.find_all('table', r = "%d" % i)
       title = currentRow[0].find_all('a')[1].text
       lwrTitle = title.lower()
       # 查找是否有全新标签
       if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
           newFlag = 1.0
       else:
           newFlag = 0.0
       # 查找是否已经标志出售，我们只收集已出售的数据
       soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
       if len(soldUnicde) != 0:
           # 解析页面获取当前价格
           soldPrice = currentRow[0].find_all('td')[4]
           priceStr = soldPrice.text
           priceStr = priceStr.replace('$','')
           priceStr = priceStr.replace(',','')
           if len(soldPrice) > 1:
               priceStr = priceStr.replace('Free shipping', '')
           sellingPrice = float(priceStr)
           # 去掉不完整的套装价格
           if  sellingPrice > origPrc * 0.5:
               retX.append([yr, numPce, newFlag, origPrc])
               retY.append(sellingPrice)
       i += 1
       currentRow = soup.find_all('table', r = "%d" % i)
#负责获取乐高的数据
def setDataCollect(retX, retY):
   scrapePage(retX, retY, './setHtml/lego8288.html', 2006, 800, 49.99)                #2006年的乐高8288,部件数目800,原价49.99
   scrapePage(retX, retY, './setHtml/lego10030.html', 2002, 3096, 269.99)                #2002年的乐高10030,部件数目3096,原价269.99
   scrapePage(retX, retY, './setHtml/lego10179.html', 2007, 5195, 499.99)                #2007年的乐高10179,部件数目5195,原价499.99
   scrapePage(retX, retY, './setHtml/lego10181.html', 2007, 3428, 199.99)                #2007年的乐高10181,部件数目3428,原价199.99
   scrapePage(retX, retY, './setHtml/lego10189.html', 2008, 5922, 299.99)                #2008年的乐高10189,部件数目5922,原价299.99
   scrapePage(retX, retY, './setHtml/lego10196.html', 2009, 3263, 249.99)                #2009年的乐高10196,部件数目3263,原价249.99
if __name__ == '__main__':
   lgX = []
   lgY = []
   setDataCollect(lgX, lgY)
   for i in range(len(lgX)):
        for j in lgX[i]:
            print(j,end = '\t')
        print(lgY[i])