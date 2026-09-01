import pandas as pd
def get_queries(n:int = 30):
    df = pd.read_csv(r"D:\enterprise_research_agent\enterprise_research_agent\evaluation\Eval_data.csv")

    df = df[["query","answer","type","source"]]

    #print(df.head())

    df = df.sample(n,random_state = 137).reset_index(drop=True)

    # print(df.shape)

    # for query in df["query"]:
    #     print(query)

    result = df.to_dict(orient="records")

    return result

# print(get_queries())