<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head profile="http://selenium-ide.openqa.org/profiles/test-case">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<link rel="selenium.base" href="http://www.nytimes.com/" />
<title>nytimes.com</title>
</head>
<body>
<table cellpadding="1" cellspacing="1" border="1">
<thead>
<tr><td rowspan="1" colspan="3">nytimes.com</td></tr>
</thead><tbody>
<tr>
	<td>open</td>
	<td>/</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//div[@class='story'][1]/descendant::a</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>css=#upNextWrapper a</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//div[@id='mostPopContentMostEmailed']/descendant::a[1]</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//div[@id='mostPopContentMostEmailed']/descendant::a[2]</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>id=NYTLogo</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>link=Health</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//div[@class='story'][1]/descendant::a</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>link=Health</td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//div[@class='story'][2]/descendant::a</td>
	<td></td>
</tr>
</tbody></table>
</body>
</html>
