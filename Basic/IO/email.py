from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders


class MyEmail(object):
    def __init__(self, user, password, server):
        self.server = {"user": user, "password": password, "name": server}

    def send(self, to, subject, text, file_path, files):
        self._send_mail(self.server, self.server['user'], to, subject, text, file_path, files)

    @classmethod
    def _send_mail(cls, server, fro, to, subject, text, file_path, files):
        assert type(server) == dict
        assert type(to) == list
        assert type(files) == list

        msg = MIMEMultipart()
        msg['From'] = fro
        msg['Subject'] = subject
        msg['To'] = COMMASPACE.join(to)  # COMMASPACE==', '
        msg['Date'] = formatdate(localtime=True)
        msg.attach(MIMEText(text))

        for file in files:
            path = file_path + file
            part = MIMEBase('application', 'octet-stream')  # 'octet-stream': binary data
            part.set_payload(open(path, 'rb').read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % file)
            msg.attach(part)
        import smtplib
        smtp = smtplib.SMTP(server['name'])
        smtp.login(server['user'], server['password'])
        smtp.sendmail(fro, to, msg.as_string())
        smtp.close()
