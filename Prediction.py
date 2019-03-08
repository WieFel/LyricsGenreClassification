from Preprocessing import preprocessing_pipeline
from FeatureExtraction import extract_feature_vector
from Preprocessing import DATA_PATH
import pickle

text = """Tell me somethin', girl
Are you happy in this modern world?
Or do you need more?
Is there somethin' else you're searchin' for?
I'm fallin'
In all the good times I find myself longin' for change
And in the bad times, I fear myself


Tell me something, boy
Aren't you tired tryin' to fill that void?
Or do you need more?
Ain't it hard keepin' it so hardcore?
I'm falling
In all the good times I find myself longing for change
And in the bad times, I fear myself
I'm off the deep end, watch as I dive in
I'll never meet the ground
Crash through the surface, where they can't hurt us
We're far from the shallow now
In the sha-ha-sha-ha-llow
In the sha-ha-sha-la-la-la-llow
In the sha-ha-sha-ha-llow
We're far from the shallow now
Oh, hahhh-oh
Ah, haaahw-woah-woah, ohhh
Ah, haaahw-woah-woah, ohhh
I'm off the deep end, watch as I dive in
I'll never meet the ground
Crash through the surface, where they can't hurt us
We're far from the shallow now
In the sha-ha-sha-ha-llow
In the sha-ha-sha-la-la-la-llow
In the sha-ha-sha-ha-llow
We're far from the shallow now"""

# ------------------------- MAIN CODE ------------------------- #
if __name__ == "__main__":
    pre = preprocessing_pipeline(text)
    print("Preprocessed: " + str(pre))
    features = extract_feature_vector(pre)
    print("Features: " + str(features))
