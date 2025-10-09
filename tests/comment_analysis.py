import loader
from analysis.sentiment.comments import *
from analysis.db import query

comments = query("select id, by, text from items where type = 'comment' and text != '' order by id desc limit 100")

config = ModelConfig(device='cpu')
analyzer = HNSentimentTransformer(config)

comments['sentiment'] = comments.apply(
    lambda row: analyzer.analyze_comment(row.id, row.text, row.by)
    , axis=1
)
