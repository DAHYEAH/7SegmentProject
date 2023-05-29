import 'dart:io';
import 'package:http_parser/http_parser.dart';
import 'package:dio/dio.dart';
import 'package:fluttertoast/fluttertoast.dart';

late final domain = "http://192.168.0.26:5001/";

// 서버로 분만사 사진 보내는 api
Future<List> uploadimg_maternity(File file)async{
  final api =domain+'api/ocrImageUpload';
  final dio = Dio();

  String fileName = "test.jpg";

  FormData _formData = FormData.fromMap({
    "files" : await MultipartFile.fromFile(file.path,
        filename: fileName, contentType : MediaType("image","jpg")),
  });

  Response response = await dio.post(
      api,
      data:_formData,
      onSendProgress: (rec, total) {
        print('Rec: $rec , Total: $total');
      }
  );
  if(response.statusCode == 200){
    resultToast("upload success");
  }
  else{
    resultToast("upload failed");
  }
  print(response.data);
  return response.data;
}

// 결과를 toast로 띄우는 함수
resultToast(String msg) {
  Fluttertoast.showToast(
      msg: msg,
      toastLength: Toast.LENGTH_LONG,
      gravity: ToastGravity.BOTTOM,
      timeInSecForIosWeb: 3,
      fontSize: 16.0
  );
}
