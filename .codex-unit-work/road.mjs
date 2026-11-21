import {createRequire} from 'node:module';
import fs from 'node:fs/promises';
const {Workbook,SpreadsheetFile,FileBlob}=createRequire('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt/dependencies/node/load.cjs')('@oai/artifact-tool');
const w=Workbook.create(), s=w.worksheets.add('Дорожная карта');
s.showGridLines=false;
const stages=[
['Данные и модель',0,6,'Steel Plates Faults — 1 941 запись, 27 признаков, 7 типов дефектов. Стратифицированное разбиение (seed 42): 50% на обучение, 15% на калибровку порогов, 15% в пул разметки и 20% на контроль (389 записей). Сравнили 6 моделей на 5 фолдах с 3 семенами. Критерий выбора — macro-F1.'],
['Каркас сервиса',7,14,'Сервис на FastAPI с четырьмя экранами: распознавание, дашборд, проверка и качество. Импорт CSV, JSON и BSON. Проверка 27 обязательных полей допускает любой порядок столбцов и игнорирует дополнительные. При ошибке сервис указывает запись и поле.'],
['Мониторинг сдвига',15,22,'KS-статистика по каждому признаку. Порог — 95-й процентиль максимума по 96 розыгрышам для партий из 30, 60, 100, 150 и 200 записей. Монитор обнаружил сдвиг яркости по 4 признакам при уверенном прогнозе модели. На 40 парах партий измерили обнаружение сдвигов и ложные тревоги.'],
['Контролёр в контуре',23,30,'Очередь учитывает разрыв между двумя наиболее вероятными классами и разнообразие записей. Контролёр подтверждает или исправляет метку. Дообучение случайного леса из 160 деревьев создаёт новую версию. Контрольная выборка исключена из обучения, предыдущие версии сохраняются.'],
['Экономика и Optuna',31,36,'Экономия определяется разницей бюджетов разметки до достижения целевого macro-F1 с учётом времени и ставки оператора. Предприятие задаёт минуты и рубли. Расчёт допускает нулевой и отрицательный результат. Optuna использована для подбора параметров XGBoost версии v0.'],
['Дашборд и упаковка',37,40,'Дашборд объединяет сравнение моделей, матрицу ошибок на контрольной выборке, KS по признакам, очередь проверки, кривые обучения и калькулятор экономии. Подготовлены тесты, Docker с непривилегированным пользователем, конфигурация деплоя и макеты для телефона и компьютера.']];
const r=a=>s.getRange(a);
const title=(a,v)=>{r(a).merge();r(a.split(':')[0]).values=[[v]]};
r('A1:H42').format.font={name:'Arial',size:11,color:'#193b49'};
for(const [c,n] of Object.entries({A:29,B:11,C:11,D:11,E:18,F:18,G:18,H:18}))r(c+'1:'+c+'42').format.columnWidth=n;
title('A1:H1','Stal-In   /   Дорожная карта');r('A1').format.font={size:17,bold:true,color:'#184759'};r('A1').format.rowHeight=32;
title('A21:H21','Следующие шаги  ·  горизонт около года');r('A21:H21').format={fill:'#164958',font:{color:'#FFFFFF',bold:true,size:12},rowHeight:26};
for(const [i,[a,t]] of [['A22:B22','1. Продакшн'],['C22:D22','2. Замкнутый контур'],['E22:F22','3. Настоящая обратная связь'],['G22:H22','4. Масштаб и предписание']].entries()){
title(a,t);r(a).format={fill:i%2?'#d2e9e4':'#e6f3ef',rowHeight:32,horizontalAlignment:'center',font:{bold:true,color:'#164958',size:10}};}
r('A25:D25').values=[['Этап','Начало, ч','Конец, ч','Интервал, ч']];title('E25:H25','Результат этапа');
r('A25:H25').format={fill:'#164958',font:{bold:true,color:'#FFFFFF'},rowHeight:23};
for(const [i,[name,a,b,detail]] of stages.entries()){let n=26+i;
r('A'+n+':C'+n).values=[[name,a,b]];r('D'+n).formulas=[['=C'+n+'-B'+n]];
title('E'+n+':H'+n,detail);r('A'+n+':H'+n).format.rowHeight=94;r('A'+n+':H'+n).format.verticalAlignment='center';r('E'+n+':H'+n).format.wrapText=true;
if(i%2===0)r('A'+n+':H'+n).format.fill='#eef5f4';}
const ch=s.charts.add('bar',{title:'Разработка прототипа по часам'});ch.title='Разработка прототипа по часам';ch.hasLegend=false;ch.setPosition('A3','I20');ch.barOptions.direction='bar';ch.barOptions.grouping='stacked';ch.barOptions.overlap=100;
for(const [name,col,color] of [['Начало','B','#ffffff'],['Работа над этапом','D','#137567']]){
const series=ch.series.add(name);series.categoryFormula="'Дорожная карта'!$A$26:$A$31";series.formula="'Дорожная карта'!$"+col+"$26:$"+col+"$31";series.fill=color;}
w.recalculate();const out='outputs/stal-roadmap';await fs.mkdir(out,{recursive:true});
await (await SpreadsheetFile.exportXlsx(w)).save(out+'/Дорожная карта.xlsx');
