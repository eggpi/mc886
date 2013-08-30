<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head profile="http://selenium-ide.openqa.org/profiles/test-case">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<link rel="selenium.base" href="http://www.youtube.com/" />
<title>youtube.com</title>
</head>
<body>
<table cellpadding="1" cellspacing="1" border="1">
<thead>
<tr><td rowspan="1" colspan="3">youtube.com</td></tr>
</thead><tbody>
<tr>
	<td>open</td>
	<td>/</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//a[@href='/channel/HCtnHdj3df7iM']</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//div[@id='gh-overviewtab']/descendant::a[starts-with(@href, '/watch')]</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//ul[@id='watch-related']/li[1]/a/</td>
	<td></td>
</tr>
<tr>
	<td>goBackAndWait</td>
	<td></td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//ul[@id='watch-related']/li[2]/a/</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//ul[@id='watch-related']/li[1]/a/</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>id=logo</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//a[@href='/channel/HCp-Rdqh3z4Uc']</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//div[@id='gh-overviewtab']/descendant::a[starts-with(@href, '/watch')]</td>
	<td></td>
</tr>
<tr>
	<td>sendKeys</td>
	<td>id=masthead-search-term</td>
	<td>mozilla</td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>css=#search-btn</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//a[@href='/user/firefoxchannel']</td>
	<td></td>
</tr>

</tbody></table>
</body>
</html>
