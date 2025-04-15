
experiment_name = "regex_without_spelling_corrector"  ##after that it could be {regex}_{correction}_{classification}_{training_type}

paths_config = {
    # directories
    "images_dir": "./images",
    "alcohol_images_dir": "./images/alcohol",
    "non_alcohol_images_dir": "./images/non_alcohol",
    "save_dir": f"./results/{experiment_name}",
    # files
    "results_file": "model_outputs.csv",
}


ocr_engine_config = {"use_angle_cls": True,
                        "lang" :'en',
                                "use_gpu":False,
                                "det_db_thresh":0.2,
                                "det_db_box_thresh":0.4,
                                "det_db_unclip_ratio":2.0}

eval_config = {
    "split_ratios": {
        "train": 0.7,
        "validation": 0.15,
        "test": 0.15
    },
    "random_seed": 42
}

experiments = [
        {
        "name": "regex_no_spelling_corrector",
        "advanced_spell_correction": False,
        "ocr_engine_settings": {
            "use_angle_cls": True,
            "lang": 'en',
            "use_gpu": False,
            "det_db_thresh": 0.2,
            "det_db_box_thresh": 0.4,
            "det_db_unclip_ratio": 2.0
        },
        "text_correction": {
            "use_autocorrector": True,
            "common_ocr_mistakes": {
                "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
                "5": "S", "6": "G", "7": "T", "8": "B", "9": "g",
                "l": "I", "vv": "w"
            }
        }
    }
        #  ,   {
        #     "name": "regex_with_spell_correction",
        #     "advanced_spell_correction": True,
        #     "ocr_engine_settings": {
        #         "use_angle_cls": True,
        #         "lang": 'en',
        #         "use_gpu": False,
        #         "det_db_thresh": 0.2,
        #         "det_db_box_thresh": 0.4,
        #         "det_db_unclip_ratio": 2.0
        #     },
        #     "text_correction": {
        #         "use_autocorrector": True,
        #         "common_ocr_mistakes": {
        #             "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
        #             "5": "S", "6": "G", "7": "T", "8": "B", "9": "g",
        #             "l": "I", "vv": "w"
        #         }
        #     }
        # },
        # {
        #     "name": "naive_ocr",
        #     "advanced_spell_correction": True,
        #     "ocr_engine_settings": {
        #         "lang": 'en',
        #         "use_gpu": False
        #     },
        #     "text_correction": {
        #         "use_autocorrector": True,
        #         "common_ocr_mistakes": {
        #             "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
        #             "5": "S", "6": "G", "7": "T", "8": "B", "9": "g",
        #             "l": "I", "vv": "w"
        #         }
        #     }
        # },
        # {
        #     "name": "without_visual_correction",
        #     "advanced_spell_correction": True,
        #     "ocr_engine_settings": {
        #         "use_angle_cls": True,
        #         "lang": 'en',
        #         "use_gpu": False,
        #         "det_db_thresh": 0.2,
        #         "det_db_box_thresh": 0.4,
        #         "det_db_unclip_ratio": 2.0
        #     },
        #     "text_correction": {
        #         "use_autocorrector": True,
        #         "common_ocr_mistakes": {
        #             "DONT": "DONT"  # Effectively disables it
        #         }
        #     }
        # }
    ]

ocr_config = {
    "OCR_CHARACTER_CORRECTIONS": {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "g",
    "l": "I", "vv": "w"
}
}

regex_classification_terms = {
        "drink categories": [
            "beer", "wine", "vodka", "rum", "whiskey", "whisky", "tequila", "bourbon",
            "brandy", "gin", "port", "sherry", "schnapps", "cider", "liqueur", "mead",
            "absinthe", "champagne", "fortified wine", "spirit", "hard seltzer", "scotch",
            "pisco", "vermouth", "sake", "soju", "pálinka", "arak", "ouzo", "shochu", "pulque",
            "singani", "raki", "jenever", "calvados", "aquavit", "slivovitz"
        ],
        "beer_types": [ #"ips"
            "lager", "ale", "stout", "pilsner", "porter", "wheat beer", "saison",
            "bock", "dunkel", "weissbier", "tripel", "dubbel", "kolsch", "gose", "amber ale",
            "brown ale", "hazy ipa", "red ale", "cream ale", "barleywine", "wild ale",
            "sour beer", "barrel-aged", "marzen", "altbier", "lambic", "kriek", "gueuze",
            "farmhouse ale", "session ipa", "milkshake ipa", "neipa", "hefeweizen", "doppelbock"
        ],
        "wine_types": [
            "merlot", "cabernet sauvignon", "pinot noir", "shiraz", "syrah", "zinfandel",
            "malbec", "sangiovese", "grenache", "tempranillo", "chardonnay", "sauvignon blanc",
            "riesling", "pinot grigio", "moscato", "prosecco", "cava", "cremant", "spumante",
            "gewurztraminer", "viognier", "chenin blanc", "verdelho", "semillon", "gruner veltliner",
            "torrontes", "carmenere", "rioja", "barolo", "brunello", "valpolicella"
        ],
        "liqueurs": [
            "baileys", "cointreau", "amaretto", "frangelico", "kahlua", "sambuca", "limoncello",
            "campari", "midori", "galliano", "drambuie", "grand marnier", "curaçao",
            "chartreuse", "benedictine", "ouzo", "anisette", "southern comfort", "fireball",
            "peach schnapps", "st-germain", "chambord", "jägermeister", "fernet", "sloe gin",
            "irish cream", "triple sec", "pastis", "pimm's", "maraschino", "tia maria"
        ],
        "brands": [
            "heineken", "budweiser", "corona", "guinness", "coors", "stella", "becks", "carlsberg",
            "modelo", "amstel", "jack daniels", "johnnie walker", "jameson", "maker's mark",
            "jim beam", "bulleit", "chivas", "macallan", "glenfiddich", "bacardi", "captain morgan",
            "malibu", "absolut", "smirnoff", "grey goose", "tanqueray", "bombay sapphire",
            "patron", "jagermeister", "peroni", "asahi", "dos equis", "rolling rock",
            "svedka", "new amsterdam", "red stripe", "blue moon", "pabst", "ninkasi",
            "robert mondavi", "kendall-jackson", "yellow tail", "barefoot", "black box", "beringer",
            "stella artois", "corona extra", "miller lite", "samuel adams", "tsingtao", "singha",
            "krombacher", "leffe", "hoegaarden", "sapporo", "kirin", "tiger", "sierra nevada",
            "lagunitas", "ballast point", "bud light", "michelob", "modelo especial"
        ],
        "popular_whiskeys": [
            "johnnie walker", "jack daniels", "jameson", "macallan", "bulleit", "jim beam",
            "wild turkey", "glenlivet", "glenfiddich", "chivas regal", "redbreast", "maker's mark",
            "bushmills", "crown royal", "yamazaki", "hibiki", "nikka", "knob creek", "booker's",
            "woodford reserve", "laphroaig", "ardbeg", "talisker", "highland park", "four roses"
        ],
        "popular_liqueurs": [
            "baileys", "kahlua", "amaretto", "frangelico", "drambuie", "midori", "cointreau",
            "curaçao", "campari", "grand marnier", "sambuca", "limoncello", "southern comfort",
            "fireball", "st-germain", "chambord", "jägermeister", "fernet", "pimm's", "tia maria"
        ],
        "cocktails": [
            "margarita", "mojito", "bloody mary", "martini", "cosmopolitan", "old fashioned",
            "manhattan", "negroni", "long island iced tea", "whiskey sour", "daiquiri",
            "mint julep", "white russian", "black russian", "pina colada", "mai tai",
            "gin and tonic", "rum and coke", "tequila sunrise", "sex on the beach",
            "espresso martini", "moscow mule", "aperol spritz", "paloma", "tom collins",
            "vesper", "sidecar", "sazerac", "dark 'n stormy", "caipirinha", "aviation",
            "corpse reviver", "zombie", "grasshopper", "french 75", "bellini", "mimosa"
        ],

        "regulatory_labels": [
            "drink responsibly", "must be 21", "not for minors", #abv
            "alcohol by volume", "enjoy in moderation", "age verification required", "21+ only",
            "no sale to minors", "not for underage", "alcohol abuse", "please drink responsibly",
            "contains ethanol", "ethyl alcohol", 
        ],
        "other": [
            "alc vol", "alc.", "alcohol", "distilled", "brewery", "winery", "vineyard", "fermented",
            "contains alcohol", "alcohol content", "produced by distillation", "shots", "chaser",
            "booze", "hooch", "firewater", "on the rocks", "neat", "shot glass", 
            "barrel-aged", "distillery", "craft beer", "abv%", "taproom", "mixer", "bitters",
            "barrel proof", "cask strength", "aged in oak", "small batch", "single malt", "blended"
        ]
}

non_alcohol_exclude = {
    "non_alcohol_exclude": [
        "non-alcoholic", "alcohol-free", "0% alcohol", "no alcohol", "contains no alcohol", "virgin cocktail",
        "mocktail", "0.0%", "zero alcohol", "soft drink", "carbonated beverage", "energy drink",
        "non-alcohol", "contains 0 alcohol", "free from alcohol", "kids beverage", "juice box",
        "kombucha", "sparkling water", "flavored water", "isotonic drink", "recovery drink", "protein shake",
        "sports drink", "ginger beer", "root beer", "herbal tonic", "diet soda", "cold brew", "iced tea",
        "n/a beverage", "dealcoholized", "alcohol-removed", "non-intoxicating", "non-alc", "na beer",
        "virgin drink", "zero proof", "sober", "juice blend", "plant-based milk", "smoothie",
        "fruit punch", "lemonade", "soda",   "club soda", "tonic water"
         "infused water", "herbal tea", "decaf"
    ]}