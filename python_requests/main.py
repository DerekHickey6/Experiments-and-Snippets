import requests
# Get, Post, Put, Patch, Delete
payload = {
    "name": "Derek",
    "age": 25
}

response = requests.post("https://httpbin.org/post", data=payload)

# Proxies
proxies = {
    "http": "139.99.237.62:80",      # must target the http
    "https": "139.99.237.62:80"      # must target https
}

response = requests.get("https://httpbin.org/get", proxies=proxies)
print(response.text)

# print(response.url)

headers = {
    "Accept": "image/png"
}
# Delay + timeout
# for _ in [1,2,3]:
#     try:
#         response = requests.get("https://httpbin.org/delay/3", timeout=3)

#     except:
#         continue
# Get image
# response = requests.get("https://httpbin.org/image", headers=headers)
print(response.text)



# with open("myimage1.jpg", "wb") as f:
#     f.write(response.content)


# if response.status_code == requests.codes.not_found:
#     print("Not Found")
# else:
#     print(response.status_code)


# response = requests.get("https://httpbin.org/get", params=params)

res_json = response.json()
del res_json['origin']
print(res_json)