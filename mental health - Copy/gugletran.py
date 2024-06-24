from googletrans import Translator
translator = Translator()
from_lang = 'en'
to_lang = 'kn'
text_to_translate = translator.translate('vchgfey5eyshrewy54',src= from_lang,dest= to_lang)
kan_text = text_to_translate.text
print(kan_text)
