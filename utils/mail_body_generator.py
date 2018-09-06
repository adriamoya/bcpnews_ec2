import json
import pprint
import pandas as pd
pp = pprint.PrettyPrinter(indent=4)

def create_email_body(data):

    print('--> Building email body ...')

    # # read data# read
    # final_articles_data = []
    # with open(data) as f:
    #     for line in f:
    #         final_articles_data.append(json.loads(line.encode('utf-8')))

    df = pd.DataFrame(data)

    msg = """\
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml">

    <head>
      <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
      <title>[SUBJECT]</title>
      <style type="text/css">
        body {
          padding-top: 0 !important;
          padding-bottom: 0 !important;
          padding-top: 0 !important;
          padding-bottom: 0 !important;
          margin: 0 !important;
          width: 100% !important;
          -webkit-text-size-adjust: 100% !important;
          -ms-text-size-adjust: 100% !important;
          -webkit-font-smoothing: antialiased !important;
        }

        .tableContent img {
          border: 0 !important;
          display: block !important;
          outline: none !important;
        }

        a {
          color: #382F2E;
        }

        p,
        h1,
        h2,
        ul,
        ol,
        li,
        div {
          margin: 0;
          padding: 0;
        }

        h1,
        h2 {
          font-weight: normal;
          background: transparent !important;
          border: none !important;
        }

        @media only screen and (max-width:480px) {

          table[class="MainContainer"],
          td[class="cell"] {
            width: 100% !important;
            height: auto !important;
          }
          td[class="specbundle"] {
            width: 100% !important;
            float: left !important;
            font-size: 13px !important;
            line-height: 17px !important;
            display: block !important;
            padding-bottom: 15px !important;
          }
          td[class="specbundle2"] {
            width: 80% !important;
            float: left !important;
            font-size: 13px !important;
            line-height: 17px !important;
            display: block !important;
            padding-bottom: 10px !important;
            padding-left: 10% !important;
            padding-right: 10% !important;
          }

          td[class="spechide"] {
            display: none !important;
          }
          img[class="banner"] {
            width: 100% !important;
            height: auto !important;
          }
          td[class="left_pad"] {
            padding-left: 15px !important;
            padding-right: 15px !important;
          }
        }

        @media only screen and (max-width:540px) {

          table[class="MainContainer"],
          td[class="cell"] {
            width: 100% !important;
            height: auto !important;
          }
          td[class="specbundle"] {
            width: 100% !important;
            float: left !important;
            font-size: 13px !important;
            line-height: 17px !important;
            display: block !important;
            padding-bottom: 15px !important;
          }
          td[class="specbundle2"] {
            width: 80% !important;
            float: left !important;
            font-size: 13px !important;
            line-height: 17px !important;
            display: block !important;
            padding-bottom: 10px !important;
            padding-left: 10% !important;
            padding-right: 10% !important;
          }

          td[class="spechide"] {
            display: none !important;
          }
          img[class="banner"] {
            width: 100% !important;
            height: auto !important;
          }
          td[class="left_pad"] {
            padding-left: 15px !important;
            padding-right: 15px !important;
          }

        }

        .contentEditable h2.big,
        .contentEditable h1.big {
          font-size: 26px !important;
        }

        .contentEditable h2.bigger,
        .contentEditable h1.bigger {
          font-size: 37px !important;
        }

        td,
        table {
          vertical-align: top;
        }

        td.middle {
          vertical-align: middle;
        }

        a.link1 {
          font-size: 13px;
          color: #27A1E5;
          line-height: 24px;
          text-decoration: none;
        }

        a {
          text-decoration: none;
        }

        .link2 {
          color: #ffffff;
          border-top: 10px solid #27A1E5;
          border-bottom: 10px solid #27A1E5;
          border-left: 18px solid #27A1E5;
          border-right: 18px solid #27A1E5;
          border-radius: 3px;
          -moz-border-radius: 3px;
          -webkit-border-radius: 3px;
          background: #27A1E5;
        }

        .link3 {
          color: #555555;
          border: 1px solid #cccccc;
          padding: 10px 18px;
          border-radius: 3px;
          -moz-border-radius: 3px;
          -webkit-border-radius: 3px;
          background: #ffffff;
        }

        .link4 {
          color: #27A1E5;
          line-height: 24px;
        }

        h2,
        h1 {
          line-height: 20px;
        }

        p {
          font-size: 14px;
          line-height: 21px;
          color: #AAAAAA;
        }

        .contentEditable li {}

        .appart p {}

        .bgItem {
          background: #ffffff;
        }

        .bgBody {
          background: #ffffff;
        }

        img {
          outline: none;
          text-decoration: none;
          -ms-interpolation-mode: bicubic;
          width: auto;
          max-width: 100%;
          clear: both;
          display: block;
          float: none;
        }
      </style>


      <script type="colorScheme" class="swatch active">
        { "name":"Default", "bgBody":"ffffff", "link":"27A1E5", "color":"AAAAAA", "bgItem":"ffffff", "title":"444444" }
      </script>


    </head>

    <body paddingwidth="0" paddingheight="0" bgcolor="#d1d3d4" style="padding-top: 0; padding-bottom: 0; padding-top: 0; padding-bottom: 0; background-repeat: repeat; width: 100% !important; -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; -webkit-font-smoothing: antialiased;"
      offset="0" toppadding="0" leftpadding="0">
      <table width="100%" border="0" cellspacing="0" cellpadding="0">
        <tbody>
          <tr>
            <td>
              <table width="600" border="0" cellspacing="0" cellpadding="0" align="center" bgcolor="#ffffff" style="font-family:helvetica, sans-serif;" class="MainContainer">
                <!-- =============== START HEADER =============== -->
                <tbody>
                  <tr>
                    <td>
                      <table width="100%" border="0" cellspacing="0" cellpadding="0">
                        <tbody>
                          <tr>
                            <td valign="top" width="20">&nbsp;</td>
                            <td>
                              <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                <tbody>
                                  <tr>
                                    <td class="movableContentContainer">
                                      <div class="movableContent" style="border: 0px; padding-top: 0px; position: relative;">

                                        <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                          <tbody>

                                            <tr>
                                              <td height="15"></td>
                                            </tr>
                                            <tr>
                                              <td>
                                                <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                                  <tbody>
                                                    <tr>
                                                      <td valign="top">
                                                        <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                                          <tbody>

                                                            <tr>
                                                              <td valign="top">
                                                                <a target='_blank' href="http://www.bluecap.com">
                                                                  <img src="https://i.imgur.com/gA9Mk2R.png" alt="Logo" title="Logo" style="max-height: 20px;">
                                                                </a>
                                                              </td>
                                                              <td width="10" valign="top">&nbsp;</td>
                                                            </tr>

                                                          </tbody>
                                                        </table>
                                                      </td>
                                                      <td valign="top" width="90" class="spechide">&nbsp;</td>
                                                    </tr>
                                                  </tbody>
                                                </table>
                                              </td>
                                            </tr>

                                          </tbody>
                                        </table>

                                      </div>
                                      <!-- =============== END HEADER =============== -->
                                      <!-- =============== START BODY =============== -->

                                      <div class="movableContent" style="border: 0px; padding-left: 10px; padding-right: 10px; padding-top: 0px; position: relative;">
                                        <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                          <tbody>
                                            <tr>
                                              <td height="40"></td>
                                            </tr>
                                            <tr>
                                              <td valign="top" width="100%">
                                                <div class='contentEditableContainer contentImageEditable'>
                                                  <div class='contentEditable' style="text-align: center;">
                                                    <img class="banner" src="https://blog.zopa.com/wp-content/uploads/2015/08/blog-newspapers.jpg" alt="Foto News" title="Foto News" width="100%" max-width=300px; border="0">
                                                </div>
                                              </td>
                                            </tr>
                                          </tbody>
                                        </table>



                                      </div>
                                      <div class="movableContent" style="border: 0px; padding-top: 0px; position: relative;">
                                        <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                          <tbody>
                                            <tr>
                                              <td height='40'></td>
                                            </tr>
                                            <tr>
                                              <td style="border: 1px solid #EEEEEE; border-radius:6px;-moz-border-radius:6px;-webkit-border-radius:6px">
                                                <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                                  <tbody>
                                                    <tr>
                                                      <td valign="top" width="40">&nbsp;</td>
                                                      <td>
                                                        <table width="100%" border="0" cellspacing="0" cellpadding="0" align="center">
                                                          <tr>
                                                            <td height='25'></td>
                                                          </tr>
                                                          <tr>
                                                            <td>
                                                              <div class='contentEditableContainer contentTextEditable'>
                                                                <div class='contentEditable' style='text-align: center;'>
                                                                  <h1 style="font-size: 20px; font-weight:600;">Bluecap Banking Breakfast</h1>
                                                                  <br>
                                                                  <p>Noticias sobre el sector financiero español</p>
                                                                </div>
                                                              </div>
                                                            </td>
                                                          </tr>
                                                          <tr>
                                                            <td height='24'></td>
                                                          </tr>
                                                        </table>
                                                      </td>
                                                      <td valign="top" width="40">&nbsp;</td>
                                                    </tr>
                                                  </tbody>
                                                </table>
                                              </td>
                                            </tr>
                                          </tbody>
                                        </table>



                                      </div>"""
    for index, row in df.iterrows():
        #print(index)
        msg += """
            <div class="movableContent" style="border: 0px; padding-top: 0px; position: relative;">

              <table width="100%" border="0" cellspacing="0" cellpadding="0">

                <tbody>

                  <tr>
                    <td height="20"></td>
                  </tr>

                  <tr>
                    <td>

                      <table width="100%" border="0" cellspacing="0" cellpadding="0">

                        <tbody style="margin:auto; width:95%;display:block;">

                          <tr width="100%">

                            <td class="specbundle" valign="top" align="center" width="100%">
                              <div class='contentEditableContainer contentImageEditable'>
                                <div class='contentEditable' style="margin:auto; width:80%; display:block;"><a href='""" + str(row.url) + """'>
                                  <img src="""
        msg += '"' + str(row.top_image) + '"'
        msg += """ alt="side image" data-default="placeholder" border="0" width="100%" style="width:100%; max-width=50px;"></a>
                                </div>
                              </div>
                            </td>

                          </tr>
                          <tr>

                            <td class="specbundle">

                              <table width="100%" cellpadding="0" cellspacing="0" align="center">
                                <tbody>

                                  <tr>
                                    <td>
                                      <div class='contentEditableContainer contentTextEditable'>
                                        <div class='contentEditable' style='text-align: left;'>
                                          <h2 style='margin-top:20px; font-size:20px; margin-bottom:0px; font-weight:550'><a style="color: #000000;" href='"""
        msg += str(row.url)
        msg += """'>"""
        msg += str(row.title)
        msg += """</h2></a>
                                          <br>
                                          <p style='margin:0px; font-size:16px;'>"""
        msg += str(row.text)[:350] + '...'
        msg += """</p>
                                          <br>
                                          <p style="margin-bottom:10px; font-size:13px; color: #000000;">Leer el artículo en:</p>

                                          <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                            <tbody>
                                              <tr>"""


        msg += "<td width='40' height='40' style='padding-right:3px;'><a href='" + str(row['url']) + "'>"
        if row['newspaper'] == 'expansion':
          msg += """<img src="https://i.imgur.com/hggCaa9.png" height="40"></a></td>"""
        elif row['newspaper'] == 'eleconomista':
          msg += """<img src="https://i.imgur.com/oOe2GPY.png" height="40"></a></td>"""
        elif row['newspaper'] == 'elconfidencial':
          msg += """<img src="https://i.imgur.com/YcHL8uv.png" height="40"></a></td>"""
        elif row['newspaper'] == 'cincodias':
          msg += """<img src="https://i.imgur.com/dwmURoR.png" height="40"></a></td>"""

        if len(row.related_articles) > 0:
          for related_article in row.related_articles:
            msg += "<td width='40' height='40' style='padding-right:3px;'><a href='" + str(related_article['url']) + "'>"
            if related_article['newspaper'] == 'expansion':
              msg += """<img src="https://i.imgur.com/hggCaa9.png" height="40"></a></td>"""
            elif related_article['newspaper'] == 'eleconomista':
              msg += """<img src="https://i.imgur.com/oOe2GPY.png" height="40"></a></td>"""
            elif related_article['newspaper'] == 'elconfidencial':
              msg += """<img src="https://i.imgur.com/YcHL8uv.png" height="40"></a></td>"""
            elif related_article['newspaper'] == 'cincodias':
              msg += """<img src="https://i.imgur.com/dwmURoR.png" height="40"></a></td>"""


        msg += """<td>
                                                  <div class='contentEditable' style='text-align: right;'>
                                                  </div>
                                                </td>
                                              </tr>

                                            </tbody>
                                          </table>

                                        </div>
                                      </div>
                                    </td>
                                  </tr>

                                </tbody>
                              </table>

                            </td>
                          </tr>

                          <tr>
                            <td height="20"></td>
                          </tr>

                        </tbody>
                      </table>

                    </div>
                    <hr style='height:1px;background:#DDDDDD;border:none;'>
        """
    msg += """<!-- =============== START FOOTER =============== -->

        <div class="movableContent" style="border: 0px; padding-top: 0px; position: relative;">
        <table width="100%" border="0" cellspacing="0" cellpadding="0">
          <tbody>
            <tr>
              <td height="48"></td>
            </tr>
            <tr>
              <td>
                <table width="100%" border="0" cellspacing="0" cellpadding="0">
                  <tbody>
                    <tr>
                      <td valign="top" width="90" class="spechide">&nbsp;</td>
                      <td>
                        <table width="100%" cellpadding="0" cellspacing="0" align="center">
                          <tr>
                            <td>
                              <div class='contentEditableContainer contentTextEditable'>
                                <div class='contentEditable' style='text-align: center;color:#AAAAAA;'>
                                  <p>
                                    Sent by La1aB0t_v17
                                    <br>
                                    BluecapLabs Technologies
                                    <br>
                                    Super Reserved &copy;
                                  </p>
                                </div>
                              </div>
                            </td>
                          </tr>
                        </table>
                      </td>
                      <td valign="top" width="90" class="spechide">&nbsp;</td>
                    </tr>
                  </tbody>
                </table>
              </td>
            </tr>
          </tbody>
        </table>


        </div>

        </td>
    </tr>
    </tbody>
    </table>
    </td>
    <td valign="top" width="20">&nbsp;</td>
    </tr>
    </tbody>
    </table>
    </td>
    </tr>
    </tbody>
    </table>
    </td>
    </tr>
    </tbody>
    </table>

    </body>

    </html>
    """
    return msg
