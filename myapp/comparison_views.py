from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from .comparison import compare,options

@require_GET
def comparison_view(request):
    return render(request,'comparison.html',{'options':options()})

@require_GET
def comparison_api(request):
    try:
        data=compare(request.GET.get('scenario','original'),int(request.GET.get('market','43000010')),int(request.GET.get('model','0')))
        response=JsonResponse(data)
    except (ValueError,TypeError) as exc:
        response=JsonResponse({'error':str(exc)},status=400)
    except RuntimeError as exc:
        response=JsonResponse({'error':str(exc)},status=409)
    except (OSError,KeyError,StopIteration):
        response=JsonResponse({'error':'The saved study is incomplete or unavailable. Check the local research record.'},status=503)
    response['Cache-Control']='no-store'
    return response
