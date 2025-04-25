import pandas as pd

def formatData(filePath, outputPath):
    with open (filePath, 'r') as f:
        lines = f.readlines()

    newAbstractStart = 0
    allAbstracts = []
    abstractTextTemplate = {'BACKGROUND': '', 'OBJECTIVE': '', 'METHODS': '', 'RESULTS': '', 'CONCLUSIONS': ''}
    abstractText = {}
    for ind, line in enumerate(lines):
        if line.startswith('###'):
            allAbstracts.append(abstractText)
            abstractText = abstractTextTemplate.copy()
            continue
        else:
            parts = line.split('\t')
            try:
                abstractText[parts[0]] += parts[1].replace('\n', '')
            except KeyError:
                continue


    allAbstracts.pop(0)
    df = pd.DataFrame(allAbstracts)
    df = df.dropna()

    # This makes sure only problem and method are used
    records = {"problem_text": [], "approach_text": []}
    for _, row in df.iterrows():
        if pd.isna(row["METHODS"]):
            continue

        if not pd.isna(row["OBJECTIVE"]):
            problem = row["OBJECTIVE"]
        elif not pd.isna(row["BACKGROUND"]):
            problem = row["BACKGROUND"]
        else:
            continue

        records["problem_text"].append(str(problem))
        records["approach_text"].append(str(row["METHODS"]))

    df = pd.DataFrame(records)

    df.to_csv(outputPath, index=False)

if __name__ == "__main__":
    formatData("../data/raw/200k_abstracts/train.txt", "../data/processed/abstracts_train.csv")
    formatData("../data/raw/200k_abstracts/test.txt", "../data/processed/abstracts_test.csv")