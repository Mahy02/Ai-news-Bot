#local
from .twit import tweeter

class TwitterHandler:
    def __init__(self):
        #twitapi is the client with all keys and secrets
        self.twitapi = tweeter()

    def tweetertweet(self, thread, llm3, source, publish_date, timestamp):
        tweets = thread.split("\n\n")
       
        # Check each tweet is under 280 chars
        for i in range(len(tweets)):
            if len(tweets[i]) > 200:    
                prompt = f"Shorten this tweet to be under 200 characters: {tweets[i]}"
                tweets[i] = llm3.predict(prompt)[:200]
        
        # Give some spacing between sentences
        tweets = [s.replace('. ', '.\n\n') for s in tweets]

        # Create two tweets for the source and publish date
        source_tweet = f"If you want to learn more, check out the original source link!: {source}"
        publish_date_tweet = f"Website/Youtube Video Published on {publish_date}"
        print(source_tweet)
        print(publish_date_tweet)
        print(timestamp)
        if timestamp != -1:
            tweet_timestamp=f"Topic Timestamp in the video: {str(timestamp)}"
        

        for tweet in tweets:
            tweet = tweet.replace('**', '')

        try:
            response = self.twitapi.create_tweet(text=tweets[0])
            id = response.data['id']
            tweets.pop(0)
            for i in tweets:
                print("tweeting: " + i)
                reptweet = self.twitapi.create_tweet(text=i, 
                                        in_reply_to_tweet_id=id, 
                                        )
                id = reptweet.data['id']
                print(id)
            # Post the source and publish date tweets
            #id=id+str(1)
            response= self.twitapi.create_tweet(text=source_tweet,in_reply_to_tweet_id=id)
            id = response.data['id']
            # id=id+1
            response= self.twitapi.create_tweet(text=publish_date_tweet,in_reply_to_tweet_id=id)
            id = response.data['id']
            if timestamp !=-1:
                 #id=id+1
                 self.twitapi.create_tweet(text=timestamp,in_reply_to_tweet_id=id)
            
            return "Tweets posted successfully"
        except Exception as e:
            return f"Error posting tweets: {e}"



