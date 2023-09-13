Search.setIndex({"docnames": ["custom_losses", "index", "installation", "license", "pytorch_partial_crf/base_crf", "pytorch_partial_crf/partial_crf", "pytorch_partial_crf/utils", "quickstart"], "filenames": ["custom_losses.md", "index.rst", "installation.md", "license.md", "pytorch_partial_crf/base_crf.rst", "pytorch_partial_crf/partial_crf.rst", "pytorch_partial_crf/utils.rst", "quickstart.md"], "titles": ["List of Changes", "Welcome to PyTorch Partial/Fuzzy CRF\u2019s documentation!", "Installation", "License", "Base CRF", "Partial/Fuzzy CRF", "Utilitaries", "Quickstart"], "terms": {"small": [0, 5], "adjust": 0, "have": 0, "been": 0, "made": 0, "origin": 0, "code": 0, "suit": 0, "my": 0, "need": [0, 4], "thi": [0, 3, 4], "i": [0, 3, 4, 5], "modul": [0, 1, 4], "anymor": 0, "rather": 0, "script": 0, "add": 0, "project": 0, "The": [0, 3], "crf": 0, "marginalcrf": 0, "remov": 0, "avoid": 0, "redund": 0, "some": 0, "useless": 0, "variabl": 0, "tensor": [0, 4, 5, 6, 7], "paramet": [0, 4, 5, 6], "declar": 0, "buth": 0, "main": 0, "goal": 0, "wa": 0, "abl": 0, "us": [0, 1, 3, 4, 5], "som": 0, "function": [0, 4, 5], "onli": [0, 2], "tend": 0, "overfit": 0, "nois": [0, 5], "default": [0, 5], "optim": 0, "mathcal": 0, "l": 0, "_": 0, "nll": [0, 5], "left": 0, "frac": 0, "exp": 0, "sum_": 0, "1": [0, 2, 4, 5, 6, 7], "e": 0, "textbf": 0, "x": 0, "y_": 0, "t": 0, "z": 0, "right": [0, 3], "unlikelihood": [0, 5], "2": [0, 2, 4, 5, 7], "regular": [0, 5], "cnll": 0, "p": 0, "y": 0, "pred": 0, "true": 0, "neq": 0, "nlu": 0, "robust": [0, 5], "3": [0, 5, 7], "margin": [0, 4], "probabl": [0, 4, 5], "each": [0, 4, 5], "token": [0, 4, 5], "q": [0, 5], "hyperparamet": [0, 5], "gce": [0, 5], "0": [0, 2, 5, 7], "welleck": [0, 5], "sean": [0, 5], "et": [0, 5], "al": [0, 5], "neural": [0, 4, 5], "text": [0, 5], "gener": [0, 5], "train": [0, 4, 5], "arxiv": [0, 5], "preprint": [0, 5], "1908": [0, 5], "04319": [0, 5], "2019": [0, 3, 5], "jiang": [0, 5], "haom": [0, 5], "name": [0, 5], "entiti": [0, 5], "recognit": [0, 5], "strongli": [0, 5], "label": [0, 4, 5, 6], "larg": [0, 5], "weakli": [0, 5], "data": [0, 4, 5], "2106": [0, 5], "08977": [0, 5], "2021": [0, 5], "zhang": [0, 5], "zhilu": [0, 5], "mert": [0, 5], "sabuncu": [0, 5], "deep": [0, 4, 5], "network": [0, 4, 5], "noisi": [0, 5], "advanc": [0, 5], "inform": [0, 5], "process": [0, 5], "system": [0, 5], "31": [0, 5], "2018": [0, 5], "4": [0, 5, 7], "meng": [0, 5], "yu": [0, 5], "distantli": [0, 5], "supervis": [0, 5], "learn": [0, 5], "languag": [0, 5], "model": [0, 4, 5], "augment": [0, 5], "self": [0, 5], "2109": [0, 5], "05003": [0, 5], "instal": 1, "depend": [1, 5], "list": [1, 4], "chang": 1, "custom": 1, "loss": [1, 5], "refer": [1, 4, 5], "base": 1, "utilitari": 1, "licens": 1, "index": [1, 4, 6], "It": 2, "torch": [2, 4, 5, 6, 7], "pip": 2, "r": 2, "requir": 2, "txt": 2, "copyright": 3, "koga": 3, "kobayashi": 3, "kajyuuen": 3, "gmail": 3, "com": 3, "permiss": 3, "herebi": 3, "grant": 3, "free": 3, "charg": 3, "ani": 3, "person": 3, "obtain": 3, "copi": 3, "softwar": 3, "associ": 3, "document": 3, "file": 3, "deal": 3, "without": 3, "restrict": 3, "includ": 3, "limit": 3, "modifi": [3, 5], "merg": 3, "publish": 3, "distribut": [3, 4], "sublicens": 3, "sell": 3, "permit": 3, "whom": 3, "furnish": 3, "do": [3, 4], "so": 3, "subject": 3, "follow": 3, "condit": [3, 4, 5], "abov": 3, "notic": 3, "shall": 3, "all": [3, 4], "substanti": 3, "portion": 3, "THE": 3, "provid": 3, "AS": 3, "warranti": 3, "OF": 3, "kind": 3, "express": 3, "OR": 3, "impli": 3, "BUT": 3, "NOT": 3, "TO": 3, "merchant": 3, "fit": 3, "FOR": 3, "A": 3, "particular": 3, "purpos": 3, "AND": 3, "noninfring": 3, "IN": 3, "NO": 3, "event": 3, "author": 3, "holder": 3, "BE": 3, "liabl": 3, "claim": 3, "damag": 3, "other": 3, "liabil": 3, "whether": 3, "an": 3, "action": 3, "contract": 3, "tort": 3, "otherwis": 3, "aris": 3, "from": [3, 4, 5, 7], "out": 3, "connect": 3, "WITH": 3, "class": [4, 5], "pytorch_partial_crf": [4, 5, 6, 7], "base_crf": 4, "basecrf": 4, "num_tag": [4, 5, 6, 7], "int": [4, 5, 6], "devic": [4, 5, 7], "liter": [4, 5], "cpu": [4, 5, 7], "cuda": [4, 5], "padding_idx": [4, 5], "none": [4, 5], "sourc": [4, 5, 6], "abstract": 4, "method": 4, "random": [4, 5], "field": [4, 5], "number": [4, 6], "possibl": 4, "tag": [4, 5, 6, 7], "count": 4, "pad": [4, 5], "one": 4, "option": [4, 5], "str": [4, 5], "wether": 4, "comput": [4, 5], "gpu": 4, "type": [4, 5, 6], "start_transit": 4, "begin": 4, "score": [4, 5], "transit": 4, "matrix": 4, "initi": 4, "valu": [4, 6], "sampl": 4, "uniform": 4, "nn": 4, "end_transit": 4, "end": 4, "xavier": 4, "": 4, "ar": 4, "lafferti": 4, "john": 4, "andrew": 4, "mccallum": 4, "fernando": 4, "cn": 4, "pereira": 4, "probabilist": 4, "segment": 4, "sequenc": 4, "2001": 4, "glorot": 4, "yoshua": 4, "bengio": 4, "understand": 4, "difficulti": 4, "feedforward": 4, "proceed": 4, "thirteenth": 4, "intern": 4, "confer": 4, "artifici": 4, "intellig": 4, "statist": 4, "jmlr": 4, "workshop": 4, "2010": 4, "forward": [4, 5], "emiss": [4, 5, 7], "floattensor": [4, 5], "longtensor": [4, 5, 6], "mask": [4, 5, 6, 7], "bytetensor": [4, 5, 6], "defin": 4, "perform": [4, 5], "everi": 4, "call": 4, "should": 4, "overridden": 4, "subclass": 4, "although": 4, "recip": 4, "pass": [4, 5], "within": 4, "instanc": 4, "afterward": 4, "instead": 4, "sinc": 4, "former": 4, "take": 4, "care": 4, "run": 4, "regist": 4, "hook": 4, "while": 4, "latter": 4, "silent": 4, "ignor": 4, "them": 4, "marginal_prob": [4, 7], "unari": [4, 5], "batch_siz": [4, 5, 6, 7], "sequence_length": [4, 5, 6, 7], "discard": [4, 5], "subword": [4, 5], "special": [4, 5], "being": [4, 5], "ad": [4, 5], "log": [4, 5], "return": [4, 5, 6], "belong": 4, "given": 4, "viterbi_decod": [4, 7], "dynam": 4, "best": 4, "best_tags_list": 4, "batch": [4, 5], "where": [4, 5, 6], "shape": 4, "partial_crf": 5, "partialcrf": [5, 7], "float": 5, "cross": 5, "entropi": 5, "7": 5, "loss_fn": [5, 7], "c_nll": 5, "chosen": 5, "classic": 5, "neg": 5, "likelihood": 5, "correct": [5, 6], "generel": 5, "contain": 5, "target": [5, 6], "mean": [5, 7], "over": 5, "mini": 5, "util": 6, "create_possible_tag_mask": 6, "creat": 6, "like": 6, "spars": 6, "ha": 6, "allow": 6, "multilabel": 6, "differ": 6, "dataset": 6, "indic": 6, "rememb": 7, "unknown": 7, "import": 7, "9": 7, "5": 7, "randn": 7, "5437": 7, "9088": 7, "4173": 7, "3075": 7, "0963": 7, "1396": 7, "0843": 7, "2068": 7, "7572": 7, "5796": 7, "4185": 7, "6221": 7, "8547": 7, "9173": 7, "9208": 7, "4390": 7, "7294": 7, "2982": 7, "4782": 7, "7222": 7, "5666": 7, "7675": 7, "3230": 7, "4046": 7, "4232": 7, "4828": 7, "8027": 7, "0995": 7, "4749": 7, "4170": 7, "5631": 7, "5672": 7, "4975": 7, "5789": 7, "9422": 7, "0219": 7, "1128": 7, "9551": 7, "0825": 7, "8257": 7, "2484": 7, "1888": 7, "6151": 7, "7292": 7, "6003": 7, "4377": 7, "2834": 7, "0981": 7, "5948": 7, "9315": 7, "4660": 7, "3846": 7, "2995": 7, "0706": 7, "3094": 7, "0249": 7, "9489": 7, "0665": 7, "0557": 7, "9480": 7, "6224": 7, "0894": 7, "3665": 7, "1289": 7, "7502": 7, "7008": 7, "5063": 7, "6002": 7, "3744": 7, "0519": 7, "4107": 7, "9092": 7, "7128": 7, "9601": 7, "0653": 7, "6548": 7, "8773": 7, "4040": 7, "2110": 7, "2022": 7, "0100": 7, "9134": 7, "2474": 7, "2166": 7, "1720": 7, "3302": 7, "0470": 7, "2935": 7, "3067": 7, "0624": 7, "randint": 7, "bernoulli": 7, "empti": 7, "uniform_": 7, "byte": 7, "dtype": 7, "uint8": 7, "0437": 7, "4929": 7, "3082": 7, "0818": 7, "0734": 7, "2032": 7, "2544": 7, "2469": 7, "2462": 7, "0493": 7, "0467": 7, "1330": 7, "2178": 7, "3913": 7, "2112": 7, "0718": 7, "5613": 7, "0633": 7, "2867": 7, "0169": 7, "1545": 7, "4207": 7, "0120": 7, "2460": 7, "1668": 7, "1221": 7, "0251": 7, "0200": 7, "8237": 7, "0091": 7, "0707": 7, "2742": 7, "2635": 7, "1047": 7, "2869": 7, "3246": 7, "2208": 7, "1100": 7, "2588": 7, "0856": 7, "6231": 7, "0761": 7, "2051": 7, "0722": 7, "0235": 7, "grad_fn": 7, "expbackward0": 7, "209": 7, "5386": 7, "meanbackward0": 7, "618": 7, "7924": 7, "0149": 7}, "objects": {"pytorch_partial_crf": [[4, 0, 0, "-", "base_crf"], [5, 0, 0, "-", "partial_crf"], [6, 0, 0, "-", "utils"]], "pytorch_partial_crf.base_crf": [[4, 1, 1, "", "BaseCRF"]], "pytorch_partial_crf.base_crf.BaseCRF": [[4, 2, 1, "", "device"], [4, 2, 1, "", "end_transitions"], [4, 3, 1, "", "forward"], [4, 3, 1, "", "marginal_probabilities"], [4, 2, 1, "", "num_tags"], [4, 2, 1, "", "start_transitions"], [4, 2, 1, "", "transitions"], [4, 3, 1, "", "viterbi_decode"]], "pytorch_partial_crf.partial_crf": [[5, 1, 1, "", "PartialCRF"]], "pytorch_partial_crf.partial_crf.PartialCRF": [[5, 3, 1, "", "forward"], [5, 2, 1, "", "q"]], "pytorch_partial_crf.utils": [[6, 4, 1, "", "create_possible_tag_masks"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:attribute", "3": "py:method", "4": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "method", "Python method"], "4": ["py", "function", "Python function"]}, "titleterms": {"list": 0, "chang": 0, "custom": [0, 7], "loss": [0, 7], "neg": [0, 7], "log": [0, 7], "likelihood": [0, 7], "correct": [0, 7], "generela": 0, "cross": [0, 7], "entropi": [0, 7], "refer": 0, "welcom": 1, "pytorch": 1, "partial": [1, 5, 7], "fuzzi": [1, 5, 7], "crf": [1, 4, 5, 7], "": 1, "document": 1, "quickstart": [1, 7], "code": 1, "about": 1, "indic": 1, "tabl": 1, "instal": 2, "depend": 2, "licens": 3, "base": 4, "utilitari": 6, "us": 7, "decod": 7, "comput": 7, "margin": 7, "probabl": 7, "forward": 7, "pass": 7, "nll": 7, "c_nll": 7, "gener": 7, "gce": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.todo": 2, "sphinx": 57}, "alltitles": {"List of Changes": [[0, "list-of-changes"]], "Custom Losses": [[0, "custom-losses"]], "Negative Log-Likelihood": [[0, "negative-log-likelihood"]], "Corrected Negative Log-Likelihood": [[0, "corrected-negative-log-likelihood"]], "Generelaized Cross-Entropy": [[0, "generelaized-cross-entropy"]], "References": [[0, "references"]], "Welcome to PyTorch Partial/Fuzzy CRF\u2019s documentation!": [[1, "welcome-to-pytorch-partial-fuzzy-crf-s-documentation"]], "Quickstart": [[1, "quickstart"], [7, "quickstart"]], "Code": [[1, "code"]], "About": [[1, "about"]], "Indices and tables": [[1, "indices-and-tables"]], "Installation": [[2, "installation"]], "Dependencies": [[2, "dependencies"]], "Install": [[2, "install"]], "License": [[3, "license"]], "Base CRF": [[4, "module-pytorch_partial_crf.base_crf"]], "Partial/Fuzzy CRF": [[5, "module-pytorch_partial_crf.partial_crf"]], "Utilitaries": [[6, "module-pytorch_partial_crf.utils"]], "Using the Partial/Fuzzy CRF": [[7, "using-the-partial-fuzzy-crf"]], "Decoding": [[7, "decoding"]], "Computing the Marginal Probabilities": [[7, "computing-the-marginal-probabilities"]], "Forward Pass with Custom Loss": [[7, "forward-pass-with-custom-loss"]], "Negative log-likelihood nll": [[7, "negative-log-likelihood-nll"]], "Corrected negative log-likelihood c_nll": [[7, "corrected-negative-log-likelihood-c-nll"]], "Generalized cross-entropy gce": [[7, "generalized-cross-entropy-gce"]]}, "indexentries": {"basecrf (class in pytorch_partial_crf.base_crf)": [[4, "pytorch_partial_crf.base_crf.BaseCRF"]], "device (pytorch_partial_crf.base_crf.basecrf attribute)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.device"]], "end_transitions (pytorch_partial_crf.base_crf.basecrf attribute)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.end_transitions"]], "forward() (pytorch_partial_crf.base_crf.basecrf method)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.forward"]], "marginal_probabilities() (pytorch_partial_crf.base_crf.basecrf method)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.marginal_probabilities"]], "module": [[4, "module-pytorch_partial_crf.base_crf"], [5, "module-pytorch_partial_crf.partial_crf"], [6, "module-pytorch_partial_crf.utils"]], "num_tags (pytorch_partial_crf.base_crf.basecrf attribute)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.num_tags"]], "pytorch_partial_crf.base_crf": [[4, "module-pytorch_partial_crf.base_crf"]], "start_transitions (pytorch_partial_crf.base_crf.basecrf attribute)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.start_transitions"]], "transitions (pytorch_partial_crf.base_crf.basecrf attribute)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.transitions"]], "viterbi_decode() (pytorch_partial_crf.base_crf.basecrf method)": [[4, "pytorch_partial_crf.base_crf.BaseCRF.viterbi_decode"]], "partialcrf (class in pytorch_partial_crf.partial_crf)": [[5, "pytorch_partial_crf.partial_crf.PartialCRF"]], "forward() (pytorch_partial_crf.partial_crf.partialcrf method)": [[5, "pytorch_partial_crf.partial_crf.PartialCRF.forward"]], "pytorch_partial_crf.partial_crf": [[5, "module-pytorch_partial_crf.partial_crf"]], "q (pytorch_partial_crf.partial_crf.partialcrf attribute)": [[5, "pytorch_partial_crf.partial_crf.PartialCRF.q"]], "create_possible_tag_masks() (in module pytorch_partial_crf.utils)": [[6, "pytorch_partial_crf.utils.create_possible_tag_masks"]], "pytorch_partial_crf.utils": [[6, "module-pytorch_partial_crf.utils"]]}})