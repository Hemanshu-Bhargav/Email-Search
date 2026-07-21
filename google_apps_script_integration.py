/***
 * Team built a Google Apps Script companion which was auto-populating template engine.
 * This is a helper file to showcase minimal vision of the full project.
 * Google Apps Script Companion for CPS 842 Email Search Project
 * Designed to interface with the Google Sheet database ("CPS 842 Project V1") 
 * populated by the Python email extraction and search backend[cite: 5].
 */

function onOpen() {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('Email Search Tools')
      .addItem('Open Search Sidebar', 'showSearchSidebar')
      .addToUi();
}

function showSearchSidebar() {
  var html = HtmlService.createHtmlOutput(
    '<div style="font-family:Arial; padding:10px;">' +
    '<h3>Email Search Companion</h3>' +
    '<input type="text" id="query" placeholder="Enter keyword..." style="width:100%; padding:8px; margin-bottom:10px;" />' +
    '<button onclick="runSearch()" style="padding:8px 12px; background:#4285f4; color:white; border:none; border-radius:4px;">Search</button>' +
    '<div id="results" style="margin-top:15px;"></div>' +
    '</div>' +
    '<script>' +
    'function runSearch() {' +
    '  var q = document.getElementById("query").value;' +
    '  google.script.run.withSuccessHandler(displayResults).searchSheetEmails(q);' +
    '}' +
    'function displayResults(res) {' +
    '  var div = document.getElementById("results");' +
    '  if(res.length === 0) { div.innerHTML = "<p>No matches found.</p>"; return; }' +
    '  var html = "<b>Found " + res.length + " matches:</b><br/>";' +
    '  res.forEach(function(r) {' +
    '    html += "<div style=\'border-bottom:1px solid #ccc; padding:8px 0;\'>";' +
    '    html += "<b>Subject:</b> " + r.subject + "<br/>";' +
    '    html += "<b>From:</b> " + r.from + "<br/>";' +
    '    html += "<b>Date:</b> " + r.date + "<br/>";' +
    '    html += "<small>" + r.snippet + "</small></div>";' +
    '  });' +
    '  div.innerHTML = html;' +
    '}' +
    '</script>'
  ).setTitle('Email Search');
  SpreadsheetApp.getUi().showSidebar(html);
}

/**
 * Queries the active Google Sheet corresponding to the Python export schema.
 * Target Sheet: "CPS 842 Project V1"[cite: 5]
 */
function searchSheetEmails(searchTerm) {
  var sheetName = "CPS 842 Project V1";[cite: 5]
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheetName);
  
  if (!sheet) {
    throw new Error("Sheet '" + sheetName + "' could not be found. Ensure Python script has exported data.");
  }
  
  var data = sheet.getDataRange().getValues();
  var results = [];
  var queryLower = searchTerm.toLowerCase();
  
  // Row 0 contains headers: ['Message-ID', 'Subject of Email', 'Date of Email', 'From', 'Body of Email']
  for (var i = 1; i < data.length; i++) {
    var row = data[i];
    var msgId = row[0];
    var subject = String(row[1]).toLowerCase();
    var date = row[2];
    var sender = row[3];
    var body = String(row[4]).toLowerCase();
    
    if (subject.indexOf(queryLower) !== -1 || body.indexOf(queryLower) !== -1) {
      results.push({
        messageId: msgId,
        subject: row[1],
        date: date,
        from: sender,
        snippet: String(row[4]).substring(0, 150) + '...'
      });
    }
  }
  
  return results;
}
