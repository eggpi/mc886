<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head profile="http://selenium-ide.openqa.org/profiles/test-case">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<link rel="selenium.base" href="http://www.google.com" />
<title>google.com</title>
</head>
<body>
<table cellpadding="1" cellspacing="1" border="1">
<thead>
<tr><td rowspan="1" colspan="3">google</td></tr>
</thead><tbody>
<tr>
	<td>open</td>
	<td>/</td>
	<td></td>
</tr>
<tr>
	<td>sendKeys</td>
	<td>id=gbqfq</td>
	<td>firefox</td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//descendant::li[@class=&quot;g&quot;][1]/descendant::h3[@class=&quot;r&quot;]/a</td>
	<td></td>
</tr>
<tr>
	<td>goBackAndWait</td>
	<td></td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//descendant::li[@class=&quot;g&quot;][2]/descendant::h3[@class=&quot;r&quot;]/a</td>
	<td></td>
</tr>
<tr>
	<td>goBackAndWait</td>
	<td></td>
	<td></td>
</tr>
<tr>
	<td>type</td>
	<td>id=gbqfq</td>
	<td></td>
</tr>
<tr>
	<td>sendKeys</td>
	<td>id=gbqfq</td>
	<td>mozilla</td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//descendant::li[@class=&quot;g&quot;][1]/descendant::h3[@class=&quot;r&quot;]/a</td>
	<td></td>
</tr>
<tr>
	<td>goBackAndWait</td>
	<td></td>
	<td></td>
</tr>
<tr>
	<td>clickAndWait</td>
	<td>//descendant::li[@class=&quot;g&quot;][2]/descendant::h3[@class=&quot;r&quot;]/a</td>
	<td></td>
</tr>
<tr>
	<td>goBackAndWait</td>
	<td></td>
	<td></td>
</tr>
</tbody></table>
</body>
</html>
