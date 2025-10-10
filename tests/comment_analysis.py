import loader
from services.crud import get_thread, expand_comment
from analysis.sentiment.comments import *
from analysis.db import query

story = query("""
select * from items
where type != 'comment'
    and deleted is not true
order by score desc limit 1
""")

thread = get_thread(story.iloc[0].id)
comments = expand_comment(thread.items[1].id)

config = ModelConfig(device='cpu')
analyzer = HNSentimentTransformer(config)

analysis = analyzer.analyze_thread(comments, skip_argument=True)

#comments['sentiment'] = comments.apply(
#    lambda row: analyzer.analyze_comment(row.id, row.text, row.by)
#    , axis=1
#)


