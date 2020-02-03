from pythainlp.tokenize import word_tokenize

def tokenize(request):
    try:
        text = request.json['text']
        engine = request.args.get('engine', 'newmm')
        keep_whitespace = request.args.get('keep_whitespace', 'true')
        keep_whitespace = keep_whitespace.lower() in ['true', 't', '1', 'y', 'yes']
        tokens = word_tokenize(text=text,
                               engine=engine,
                               keep_whitespace=keep_whitespace)
        return {'success': True,
                'results': tokens}
    except Exception as e:
        return {'success': False,
                'message': 'Exception: ' + str(e)}










# def normalize(request):
#     try:
#         text = request.json['text']
#         return {'success': True,
#                 'results': pythainlp.util.normalize(text)}
#     except Exception as e:
#         return {'success': False,
#                 'message': 'Exception: ' + str(e)}


# def spell(request):
#     try:
#         text = request.json['text']
#         return {'success': True,
#                 'results': pythainlp.spell(text)}
#     except Exception as e:
#         return {'success': False,
#                 'message': 'Exception: ' + str(e)}


# def pos_tag(request):
#     try:
#         tokens = request.json['tokens']
#         return {'success': True,
#                 'results': pythainlp.tag.pos_tag(tokens)}
#     except Exception as e:
#         return {'success': False,
#                 'message': 'Exception: ' + str(e)}