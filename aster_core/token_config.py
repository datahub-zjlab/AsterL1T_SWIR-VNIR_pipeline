import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


token = 'eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InppeWFuZ3pqbGFiIiwiZXhwIjoxNzIwODUxNzc1LCJpYXQiOjE3MTU2Njc3NzUsImlzcyI6IkVhcnRoZGF0YSBMb2dpbiJ9.CCK8s-GzF0K71_1DoSA7ZFBvNUrhfFTa3L3E9_Q-xFAh9HA2hNXm8DPZFXB2rKqeCUos4Ft4u4TcMVqHyZEyUjrjYY_0a4aZYpEV5pmIaHjrvkCJ5-kde9WptP9aDY98IgwZOxceowt0F8uE-QkrQpmGVYw5b5CuJuBgxL1GAPykJsbEkiEFaEyXsBGIVh9faZVy8IkS8e4S-5UOIpgdaklKvjhydpITm_edM1y7zsm52mVoWQat3Mrt92FAck3hQSQNSfuobZnllgMuVYh7BgoHpVSItOhK1o7W1RI3h5r48qkxTPGpQYx9CGJqHAfEu3JJIGktY_jjwYEwKnvInA'
accessKeyId = 'E3sBq1Ron04A7TDS'
accessKeySecret = '70WbqcV7GZphRMVq3YErfmqX3Qlrz5'

params = {
        "dbname": "asterl1tmeta",
        "user": "postgres",
        "password": "p0st.9res",
        "host": "10.15.25.124",
        "port": "5432"
    }