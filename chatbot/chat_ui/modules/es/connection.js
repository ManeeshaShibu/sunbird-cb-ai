var elasticsearch=require('elasticsearch');


var client = new elasticsearch.Client( {  
  hosts: [
    //'http://'+process.env.APP_ES_USERNAME+':'+process.env.APP_ES_PASSWORD+'@'+process.env.APP_ES_HOST + ':' + process.env.APP_ES_PORT
    'http://' + process.env.APP_ES_HOST + ':' + process.env.APP_ES_PORT
  ]
});

console.log('http://'+process.env.APP_ES_HOST + ':' + process.env.APP_ES_PORT)
module.exports = client;
