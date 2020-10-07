# import pandas as pd
# import pandas._testing as tm
# import pytest

# from techminer import DataFrame

# # @pytest.fixture
# # def testdf():
# #     return pd.DataFrame(
# #         {
# #             "Authors": "author 0,author 1;author 2;author 2,author 3;author 3;author 4,author 5;author 5,author 5".split(
# #                 ";"
# #             ),
# #             "Author (s) ID": "id0;id1,id2,id2;id3,id3,id4;id5,id6;id7".split(","),
# #             "ID": list(range(6)),
# #             "Global_Citations": list(range(6)),
# #         }
# #     )


# def test_disambiguate_authors():

#     testdf = pd.DataFrame(
#         {
#             "Authors": "author 0;author 0;author 0,author 0,author 0".split(","),
#             "Author(s) ID": "0;1;2,3,4".split(","),
#             "ID": list(range(3)),
#         }
#     )

#     expected = pd.DataFrame(
#         {
#             "Authors": "author 0;author 0(1);author 0(2),author 0(3),author 0(4)".split(
#                 ","
#             ),
#             "Author(s) ID": "0;1;2,3,4".split(","),
#             "ID": list(range(3)),
#         }
#     )

#     result = DataFrame(testdf).disambiguate_authors()
#     print(result)
#     tm.assert_frame_equal(result, expected)
