import paramiko
from sshtunnel import SSHTunnelForwarder

with SSHTunnelForwarder(
    ("10.0.115.34", 443),
    ssh_username="",
    ssh_pkey="/var/ssh/rsa_key",
    ssh_private_key_password="secret",
    remote_bind_address=("10.4.52.28", 22),
    local_bind_address=('0.0.0.0', 10022)
) as tunnel:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('127.0.0.1', 5000)
    # do some operations with client session
    client.close()

print('FINISH!')
