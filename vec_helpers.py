# get_embedding = lambda word, decade: model.linear(torch.cat([torch.tensor(word_em[word]), torch.tensor(year_covar[decade])], 0))
# get_dev = lambda word: (torch.tensor(word_var[word]).exp() + 1).log()

