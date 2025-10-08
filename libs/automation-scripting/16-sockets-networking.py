"""
16-sockets-networking.py

Beginner's guide to networking, sockets, IP, and geolocation in Python.

Overview:
---------
Networking lets your Python programs communicate over the internet or local networks.
You can use sockets for low-level network communication, requests for HTTP, and APIs for geolocation and IP info.

Main Topics:
------------
- Sockets: Low-level network communication (TCP/UDP)
- IP address: Identifies devices on a network
- Geolocation: Find location info from IP
- HTTP requests: Communicate with web servers/APIs

Examples:
---------
"""

# 1. Get your local IP address
import socket
hostname = socket.gethostname()
local_ipv4 = socket.gethostbyname(hostname)
print("Local IPv4 address:", local_ipv4)

# 2. Simple TCP server and client (run in separate scripts or threads)
# --- TCP Server ---
# import socket
# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(('localhost', 12345))
# server.listen(1)
# print("Server listening...")
# conn, addr = server.accept()
# print("Connected by", addr)
# data = conn.recv(1024)
# print("Received:", data.decode())
# conn.sendall(b"Hello from server!")
# conn.close()
# server.close()

# --- TCP Client ---
# import socket
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect(('localhost', 12345))
# client.sendall(b"Hello from client!")
# data = client.recv(1024)
# print("Received:", data.decode())
# client.close()

# 3. Get your public IP address using requests
import requests
public_ip = requests.get("https://api.ipify.org").text
print("Public IP address:", public_ip)

# 4. Geolocate your IP address using an API
geo_url = f"https://ipinfo.io/{public_ip}/json"
geo_data = requests.get(geo_url).json()
print("Geolocation info:", geo_data)

# 5. Get network info (all IPs on your machine)
ips = socket.gethostbyname_ex(hostname)
print("All IPs for this host:", ips)

# 6. Ping a server (check if reachable)
import subprocess
response = subprocess.run(["ping", "google.com", "-c", "2"], capture_output=True, text=True)
print("Ping output:\n", response.stdout)

"""
Tips:
-----
- Sockets are for low-level networking (chat apps, custom protocols).
- Use requests for web APIs and HTTP communication.
- Geolocation APIs (like ipinfo.io) provide location, city, country, and more from IP.
- Always handle exceptions and errors in networking code.
- For more on sockets: https://docs.python.org/3/library/socket.html
- For requests: https://docs.python-requests.org/en/latest/
- For geolocation APIs: https://ipinfo.io/developers
- Networking can be complex—start with simple examples and build up!
"""