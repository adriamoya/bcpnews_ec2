import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class EmailSender:

    def __init__(self, address, password):
        self.address = address
        self.password = password

        if 'gmail' in self.address:
            self.server_location = 'smtp.gmail.com'
            self.port = 587
        elif ('live' in address) or ('hotmail' in address):
            self.server_location = 'smtp.live.com'
            self.port = 587
        else:
            self.server_location = 'smtp.gmail.com'
            self.port = 587

    def send_mail(self, to, subject, body, option='html', _bcc=False):
        msg = MIMEMultipart()
        msg['From'] = "Bluecap Banking Breakfast <" + self.address + ">"
        msg['To'] = ", ".join(to)
        msg['Subject'] = subject
        if _bcc:
            bcc = ['enric.gilabert@nemuru.com', 'pere.monras@nemuru.com', 'cmoyag4@gmail.com']
        else:
            bcc = ['']
        cc = ['']
        body = body

        toaddrs = to + cc + bcc
        print(toaddrs)

        msg.attach(MIMEText(body, option))

        server = smtplib.SMTP(self.server_location, self.port)
        server.starttls()
        server.login(self.address, self.password)
        text = msg.as_string()
        server.sendmail(self.address, toaddrs, text)
        server.quit()

    def send_mail_with_attach(self, to, subject, body, filename, filepath):
        msg = MIMEMultipart()
        msg['From'] = "WhiteHat GilaBot <" + self.address + ">"
        msg['To'] = ", ".join(to)
        msg['Subject'] = subject
        body = body

        msg.add_header('Content-Disposition', 'attachment', filename='./utils/logos/eleconomista.png')

        msg.attach(MIMEText(body, 'plain'))

        filename = filename
        attachment = open(filepath, "rb")

        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

        msg.attach(part)

        server = smtplib.SMTP(self.server_location, self.port)
        server.starttls()
        server.login(self.address, self.password)
        text = msg.as_string()
        server.sendmail(self.address, [to, None, ['egilaber@bluecap.com']], text)
        server.quit()
