import time
import boto3


# dynamoDBに接続
dynamodb = boto3.resource( 'dynamodb',
    aws_access_key_id = 'aws_access_key_id',
    aws_secret_access_key = 'aws_secret_access_key',
    region_name = 'region_name'
)
queue  = dynamodb.Table('roca-api-dynamodb-eval-queue')
result = dynamodb.Table('roca-api-dynamodb-eval-result')



def lambda_handler(event, context):

    # 応答するデータの格納場所
    results = []
    timestamps = []

    # 文章をqueueに追加
    with queue.batch_writer() as batch:
        for text in event['texts']:

            # timestampを発行
            time.sleep(0.00000001)
            timestamp = str(time.time())
            timestamps.append(timestamp)
            
            # queueに追加
            batch.put_item(Item={
                'timestamp': timestamp,
                'text': text
            })

    # resultを探索
    for timestamp in timestamps:
        _result = search(timestamp)
        results.append({
            'text': _result['text'],
            'score': {
                'neutral': _result['score'][0],
                'slander': _result['score'][1],
                'sarcasm': _result['score'][2],
                'sexual':  _result['score'][3],
                'spam':    _result['score'][4],
                'divulgation': _result['score'][5]
            }
        })

    return {
        'statusCode': 200,
        'body': results
    }



# 推論結果の探索
def search(timestamp):
    
    # 推論結果を検索
    _result = result.get_item(Key={
        'timestamp': timestamp
    })
    
    # データが見つかれば終了
    if 'Item' in _result:
        return {
            'text': _result['Item']['text'],
            'score': _result['Item']['score']
        }

    # 見つかるまで探索
    time.sleep(0.1)
    return search(timestamp)



# ----- テスト用処理（本番時削除） ----- #
texts = []
for i in range(20):
    texts.append('こんにちは')

event = {
    'texts': texts
}

output = lambda_handler(event, None)
print(output)
print(len(output['body']))
