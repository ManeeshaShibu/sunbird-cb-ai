var elasticsearch=require('elasticsearch');


var client = new elasticsearch.Client( {  
  hosts: [
    //'http://'+process.env.APP_ES_USERNAME+':'+process.env.APP_ES_PASSWORD+'@'+process.env.APP_ES_HOST + ':' + process.env.APP_ES_PORT
    'http://' + process.env.APP_ES_HOST || '192.168.3.211'  + ':' + process.env.APP_ES_PORT || '9200'
  ]
});

console.log('http://' + process.env.APP_ES_HOST || '192.168.3.211'  + ':' + process.env.APP_ES_PORT || '9200')
module.exports = client;
