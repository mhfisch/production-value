$(document).ready(function(){
  console.log('document is ready');

  // const response =  $.ajax('/pingtable',{
  //   method: "post",
  //   contentType: "application/json"
  // })
  // console.log(response)
  // $('#pingname').val(response.pingname)

  $('#getdata').click(async function(){
    console.log('data button clicked');

    const response = await $.ajax('/getdata',{
      method: "post",
      contentType: "application/json"
    })
    console.log(response)
    $('#myname').val(response.myname)
  })

  $('#pingtable').click(async function(){
    console.log('DB button clicked');

    const response = await $.ajax('/pingtable',{
      method: "post",
      contentType: "application/json"
    })
    console.log(response)
    $('#pingname0').val(response.pingname0)
    $('#pingthreat0').val(response.pingthreat0)
    $('#pingid0').val(response.pingid0)
    $('#pingname1').val(response.pingname1)
    $('#pingthreat1').val(response.pingthreat1)
    $('#pingid1').val(response.pingid1)
    $('#pingname2').val(response.pingname2)
    $('#pingthreat2').val(response.pingthreat2)
    $('#pingid2').val(response.pingid2)
    $('#pingname3').val(response.pingname3)
    $('#pingthreat3').val(response.pingthreat3)
    $('#pingid3').val(response.pingid3)
    $('#pingname4').val(response.pingname4)
    $('#pingthreat4').val(response.pingthreat4)
    $('#pingid4').val(response.pingid4)
  })

  $('#inference').click(async function(){
    console.log('button was clicked');

    const cylinders = parseFloat($('#cylinders').val());
    const horsepower = parseFloat($('#horsepower').val());
    const weight = parseFloat($('#weight').val());
    const data = {
      cylinders,
      horsepower,
      weight
    }
    console.log(data)

    const response = await $.ajax('/inference',{
      data: JSON.stringify(data),
      method: "post",
      contentType: "application/json"
    })
    console.log(response)
    $('#mpg').val(response.prediction)

  })
  $('#scatter-button').click(async function(){
    console.log($('#graph1')[0])
    console.log('scatter button clicked')
    const response = await $.ajax('/plot')
    console.log(response)
    const mpg = response.map(a => a[0])
    const weight = response.map(a => a[1])
    console.log(mpg)
    const trace = [{
      x:weight,
      y:mpg,
      mode:'markers',
      type:'scatter'
    }];
    const layout = {
      xaxis:{
        title:'Weight'
      },
      yaxis:{
        title:'mpg'
      },
      title:'Scatter MPG vs Weight',
      width:700,
      height:300
    }
    Plotly.plot($('#graph1')[0],trace,layout)
    $('#scatter-button').hide()
  })

})
