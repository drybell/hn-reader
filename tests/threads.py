import loader
from services.crud import get_thread, expand_comment
from analysis.db import query
from analysis.sentiment.threads import ThreadDebateAnalyzer

story = query("""
select * from items
where type != 'comment'
    and deleted is not true
order by score desc limit 1
""")

id = story.id.iloc[0]
root_thread = get_thread(id)
comment_thread = expand_comment(root_thread.items[1].id)

a = ThreadDebateAnalyzer()
analysis = a.analyze_thread(comment_thread)
