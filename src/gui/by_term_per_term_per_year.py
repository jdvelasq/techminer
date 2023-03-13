#     def summarize_by_term_per_term_per_year(
#         self, column_IDX, column_COL, sep_IDX=None, sep_COL=None, keywords=None
#     ):
#         """Computes the number of documents and citations by term per term by year.

#         Args:
#             column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
#             sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
#             column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
#             sep_COL (str): Character used as internal separator for the elements in the column_COL.
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             DataFrame.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     {
#         ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#         ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#         ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
#         ...          "Cited_by": list(range(10,16)),
#         ...          "ID": list(range(6)),
#         ...     }
#         ... )
#         >>> df
#            Year                     Authors Author Keywords  Global_Citations   ID
#         0  2010  author 0;author 1;author 2           w0;w1        10   0
#         1  2010                    author 0              w0        11   1
#         2  2011                    author 1              w1        12   2
#         3  2011                    author 3        w5;w3;w4        13   3
#         4  2012                    author 4              w5        14   4
#         5  2014                    author 4              w3        15   5

#         >>> DataFrame(df).summarize_by_term_per_term_per_year('Authors', 'Author Keywords')
#              Authors Author Keywords  Year  Global_Citations   Num Documents      ID
#         0   author 0              w0  2010        21              2  [0, 1]
#         1   author 0              w1  2010        10              1     [0]
#         2   author 1              w0  2010        10              1     [0]
#         3   author 1              w1  2010        10              1     [0]
#         4   author 2              w0  2010        10              1     [0]
#         5   author 2              w1  2010        10              1     [0]
#         6   author 1              w1  2011        12              1     [2]
#         7   author 3              w3  2011        13              1     [3]
#         8   author 3              w4  2011        13              1     [3]
#         9   author 3              w5  2011        13              1     [3]
#         10  author 4              w5  2012        14              1     [4]
#         11  author 4              w3  2014        15              1     [5]

#         >>> keywords = Keywords(['author 1', 'author 2', 'author 3', 'w1', 'w3'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).summarize_by_term_per_term_per_year('Authors', 'Author Keywords', keywords=keywords)
#             Authors Author Keywords  Year  Global_Citations   Num Documents   ID
#         0  author 1              w1  2010        10              1  [0]
#         1  author 2              w1  2010        10              1  [0]
#         2  author 1              w1  2011        12              1  [2]
#         3  author 3              w3  2011        13              1  [3]

#         """

#         data = DataFrame(
#             self[[column_IDX, column_COL, "Year", "Cited_by", "ID"]]
#         ).explode(column_IDX, sep_IDX)
#         data = DataFrame(data).explode(column_COL, sep_COL)
#         data["Num_Documents"] = 1
#         result = data.groupby([column_IDX, column_COL, "Year"], as_index=False).agg(
#             {"Cited_by": np.sum, "Num_Documents": np.size}
#         )
#         result = result.assign(
#             ID=data.groupby([column_IDX, column_COL, "Year"])
#             .agg({"ID": list})
#             .reset_index()["ID"]
#         )
#         result["Cited_by"] = result["Cited_by"].map(lambda x: int(x))
#         if keywords is not None:
#             if keywords._patterns is None:
#                 keywords = keywords.compile()
#             result = result[result[column_IDX].map(lambda w: w in keywords)]
#             result = result[result[column_COL].map(lambda w: w in keywords)]
#         result.sort_values(
#             ["Year", column_IDX, column_COL,], ascending=True, inplace=True
#         )
#         return result.reset_index(drop=True)

#     def documents_by_terms_per_terms_per_year(
#         self, column_IDX, column_COL, sep_IDX=None, sep_COL=None, keywords=None
#     ):
#         """Computes the number of documents by term per term per year.

#         Args:
#             column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
#             sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
#             column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
#             sep_COL (str): Character used as internal separator for the elements in the column_COL.
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             DataFrame.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     {
#         ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#         ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#         ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
#         ...          "Cited_by": list(range(10,16)),
#         ...          "ID": list(range(6)),
#         ...     }
#         ... )
#         >>> df
#            Year                     Authors Author Keywords  Global_Citations   ID
#         0  2010  author 0;author 1;author 2           w0;w1        10   0
#         1  2010                    author 0              w0        11   1
#         2  2011                    author 1              w1        12   2
#         3  2011                    author 3        w5;w3;w4        13   3
#         4  2012                    author 4              w5        14   4
#         5  2014                    author 4              w3        15   5

#         >>> DataFrame(df).documents_by_terms_per_terms_per_year('Authors', 'Author Keywords')
#              Authors Author Keywords  Year  Num Documents      ID
#         0   author 0              w0  2010              2  [0, 1]
#         1   author 0              w1  2010              1     [0]
#         2   author 1              w0  2010              1     [0]
#         3   author 1              w1  2010              1     [0]
#         4   author 2              w0  2010              1     [0]
#         5   author 2              w1  2010              1     [0]
#         6   author 1              w1  2011              1     [2]
#         7   author 3              w3  2011              1     [3]
#         8   author 3              w4  2011              1     [3]
#         9   author 3              w5  2011              1     [3]
#         10  author 4              w5  2012              1     [4]
#         11  author 4              w3  2014              1     [5]

#         >>> keywords = Keywords(['author 1', 'author 2', 'author 3', 'w1', 'w3'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).documents_by_terms_per_terms_per_year('Authors', 'Author Keywords', keywords=keywords)
#             Authors Author Keywords  Year  Num Documents   ID
#         0  author 1              w1  2010              1  [0]
#         1  author 2              w1  2010              1  [0]
#         2  author 1              w1  2011              1  [2]
#         3  author 3              w3  2011              1  [3]

#         """

#         result = self.summarize_by_term_per_term_per_year(
#             column_IDX, column_COL, sep_IDX, sep_COL, keywords
#         )
#         result.pop("Cited_by")
#         result.sort_values(
#             ["Year", column_IDX, column_COL],
#             ascending=[True, True, True],
#             inplace=True,
#         )
#         return result.reset_index(drop=True)

#     def citations_by_terms_per_terms_per_year(
#         self, column_IDX, column_COL, sep_IDX=None, sep_COL=None, keywords=None
#     ):
#         """Computes the number of citations by term per term per year.

#         Args:
#             column_IDX (str): the column to explode. Their terms are used in the index of the result dataframe.
#             sep_IDX (str): Character used as internal separator for the elements in the column_IDX.
#             column_COL (str): the column to explode. Their terms are used in the columns of the result dataframe.
#             sep_COL (str): Character used as internal separator for the elements in the column_COL.
#             keywords (Keywords): filter the result using the specified Keywords object.

#         Returns:
#             DataFrame.

#         Examples
#         ----------------------------------------------------------------------------------------------

#         >>> import pandas as pd
#         >>> df = pd.DataFrame(
#         ...     {
#         ...          "Year": [2010, 2010, 2011, 2011, 2012, 2014],
#         ...          "Authors": "author 0;author 1;author 2,author 0,author 1,author 3,author 4,author 4".split(","),
#         ...          "Author Keywords": "w0;w1,w0,w1,w5;w3;w4,w5,w3".split(','),
#         ...          "Cited_by": list(range(10,16)),
#         ...          "ID": list(range(6)),
#         ...     }
#         ... )
#         >>> df
#            Year                     Authors Author Keywords  Global_Citations   ID
#         0  2010  author 0;author 1;author 2           w0;w1        10   0
#         1  2010                    author 0              w0        11   1
#         2  2011                    author 1              w1        12   2
#         3  2011                    author 3        w5;w3;w4        13   3
#         4  2012                    author 4              w5        14   4
#         5  2014                    author 4              w3        15   5

#         >>> DataFrame(df).citations_by_terms_per_terms_per_year('Authors', 'Author Keywords')
#              Authors Author Keywords  Year  Global_Citations       ID
#         0   author 0              w0  2010        21  [0, 1]
#         1   author 0              w1  2010        10     [0]
#         2   author 1              w0  2010        10     [0]
#         3   author 1              w1  2010        10     [0]
#         4   author 2              w0  2010        10     [0]
#         5   author 2              w1  2010        10     [0]
#         6   author 1              w1  2011        12     [2]
#         7   author 3              w3  2011        13     [3]
#         8   author 3              w4  2011        13     [3]
#         9   author 3              w5  2011        13     [3]
#         10  author 4              w5  2012        14     [4]
#         11  author 4              w3  2014        15     [5]

#         >>> keywords = Keywords(['author 1', 'author 2', 'author 3', 'w1', 'w3'])
#         >>> keywords = keywords.compile()
#         >>> DataFrame(df).citations_by_terms_per_terms_per_year('Authors', 'Author Keywords', keywords=keywords)
#             Authors Author Keywords  Year  Global_Citations    ID
#         0  author 1              w1  2010        10  [0]
#         1  author 2              w1  2010        10  [0]
#         2  author 1              w1  2011        12  [2]
#         3  author 3              w3  2011        13  [3]


#         """

#         result = self.summarize_by_term_per_term_per_year(
#             column_IDX, column_COL, sep_IDX, sep_COL, keywords
#         )
#         result.pop("Num_Documents")
#         result.sort_values(
#             ["Year", column_IDX, column_COL],
#             ascending=[True, True, True],
#             inplace=True,
#         )
#         return result.reset_index(drop=True)
