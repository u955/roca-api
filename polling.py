### --- init
import time
from decimal import Decimal

from transformers import AutoModel, AutoTokenizer
from torch import cuda
import torch
import boto3


# dynamoDBに接続
dynamodb = boto3.resource( 'dynamodb',
    aws_access_key_id = 'aws_access_key_id',
    aws_secret_access_key = 'aws_secret_access_key',
    region_name = 'region_name'
)
queue  = dynamodb.Table('roca-api-dynamodb-eval-queue')
result = dynamodb.Table('roca-api-dynamodb-eval-result')



### --- define
class BERTClass(torch.nn.Module):
    def __init__(self, pretrained, drop_rate, otuput_size):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained)
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, otuput_size)  # BERTの出力に合わせて768次元を指定

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask, return_dict=False)
        out = self.fc(self.drop(out))
        return out


def get(pretrained):
    # パラメータの設定
    DROP_RATE = 0.4
    OUTPUT_SIZE = 6
    
    # 学習済みモデルの読込み
    filepath = F'https://roca-s3.s3.amazonaws.com/roca-model/bert-slander-class_{device}.pth'
    model = BERTClass(pretrained, DROP_RATE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
    return model


def eval(texts):
    _ids  = []
    _mask = []

    for text in texts:
        # 文章をエンコード
        inputs = tokenizer.encode_plus(
            text = text,
            add_special_tokens = True,
            max_length = 128,
            truncation = True,
            padding = 'max_length'
        )
        # _ids, _maskに追加
        _ids .append(inputs['input_ids'])
        _mask.append(inputs['attention_mask'])

    # tensorの作成
    ids  = torch.LongTensor(_ids ).to(device)
    mask = torch.LongTensor(_mask).to(device)

    # 推論開始
    results = model(ids, mask)
    return results



### --- main
def main():
    # queueを全件取得
    items = queue.scan(Limit=30)

    # itemが無ければ処理しない
    if items['Count'] == 0:
        return print('no item.')

    # 取得したデータの格納場所
    timestamp = []
    texts = []

    # データの取り出し
    for item in items['Items']:
        timestamp.append(item['timestamp'])
        texts.append(item['text'])

    # 文章の推論
    scores = eval(texts).tolist()

    
    # 推論結果をresultに追加
    with result.batch_writer() as batch:
        for i in range(len(scores)):

            # Decimal型に変換
            _score = list(map(Decimal, scores[i]))
            print(_score)

            # resultに追加
            batch.put_item(Item={
                'timestamp': timestamp[i],
                'text': texts[i],
                'score': _score
            })

    # queueから削除
    with queue.batch_writer() as batch:
        for i in range(len(scores)):

            batch.delete_item(Key={
                'timestamp': timestamp[i]
            })



### --- loop
def loop():
    # 1秒待機後に再実行
    while True:
        main()
        time.sleep(1)



### --- controller
# deviceの設定(GPUが使えない場合cpuとする)
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

# tokenizerの設定
pretrained = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer  = AutoTokenizer.from_pretrained(pretrained)

# モデルの取得
model = get(pretrained).to(device)
model.eval()

# 実行開始
loop()
