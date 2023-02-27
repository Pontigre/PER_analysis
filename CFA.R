#load libraries
library(lavaan)

# FOR CFA TO KOUROS AND ABRAMI 2006
dat <- read.csv(file='ExportedFiles/CFA_file.csv')
path <-'
f1 =~ When.I.work.in.a.group..I.do.higher.quality.work. + The.material.is.easier.to.understand.when.I.work.with.other.students. + My.group.members.help.explain.things.that.I.do.not.understand. + I.feel.working.in.groups.is.a.waste.of.time. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students.
f2 =~ My.group.members.respect.my.opinions. + My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.do.not.care.about.my.feelings. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + When.I.work.in.a.group..I.am.able.to.share.my.ideas.
f3 =~ Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful. + We.cannot.complete.the.assignment.unless.everyone.contributes. + I.let.the.other.students.do.most.of.the.work. + I.also.learn.when.I.teach.the.material.to.my.group.members. + I.learn.to.work.with.students.who.are.different.from.me.
f4 =~ I.become.frustrated.when.my.group.members.do.not.understand.the.material. + When.I.work.with.other.students..we.spend.too.much.time.talking.about.other.things. + I.have.to.work.with.students.who.are.not.as.smart.as.I.am.
'

dat <- read.csv(file='ExportedFiles/CFA_file.csv')
model <-cfa(path,data=dat)
sink("SAGE_CFA.txt")
print(summary(model))
print(fitMeasures(model))
sink()

dat <- read.csv(file='ExportedFiles/CFA_file_M.csv')
model <-cfa(path,data=dat)
sink("SAGE_CFA_M.txt")
print(summary(model))
print(fitMeasures(model))
sink()

dat <- read.csv(file='ExportedFiles/CFA_file_W.csv')
model <-cfa(path,data=dat)
sink("SAGE_CFA_W.txt")
print(summary(model))
print(fitMeasures(model))
sink()

dat <- read.csv(file='ExportedFiles/CFA_file_MW.csv')
model <-cfa(path,data=dat)
sink("SAGE_CFA_MW.txt")
print(summary(model))
print(fitMeasures(model))
sink()

path_red <-'
f1 =~ When.I.work.in.a.group..I.do.higher.quality.work. + My.group.members.help.explain.things.that.I.do.not.understand. + I.feel.working.in.groups.is.a.waste.of.time. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students.
f2 =~ My.group.members.respect.my.opinions. + My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.do.not.care.about.my.feelings. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + When.I.work.in.a.group..I.am.able.to.share.my.ideas.
f3 =~Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful. + We.cannot.complete.the.assignment.unless.everyone.contributes. + I.let.the.other.students.do.most.of.the.work. + I.also.learn.when.I.teach.the.material.to.my.group.members. + I.learn.to.work.with.students.who.are.different.from.me.
f4 =~ I.become.frustrated.when.my.group.members.do.not.understand.the.material. + I.have.to.work.with.students.who.are.not.as.smart.as.I.am.
'
model_red <-cfa(path_red,data=dat)
summary(model_red)
fitMeasures(model_red)

sink("SAGE_CFA2.txt")
print(summary(model_red))
print(fitMeasures(model_red))
sink()

# FOR EFA
#import data
data <- read.csv(file='ExportedFiles/CFA_full.csv')
print(colnames(data))

path2 <-'
f1 =~ When.I.work.in.a.group..I.do.higher.quality.work. + My.group.members.help.explain.things.that.I.do.not.understand. + When.I.work.in.a.group..I.am.able.to.share.my.ideas. + The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + My.group.members.respect.my.opinions. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f2 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
'
model2 <-cfa(path2,data=data)
summary(model2, model2.measures=TRUE)
fitMeasures(model2)

sink("EFA_n2.txt")
print(summary(model2))
print(fitMeasures(model2))
sink()

path3 <-'
f1 =~ My.group.members.help.explain.things.that.I.do.not.understand. + When.I.work.in.a.group..I.am.able.to.share.my.ideas.+My.group.members.respect.my.opinions. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f2 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.material.is.easier.to.understand.when.I.work.with.other.students.+The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
'
model3 <-cfa(path3,data=data)
summary(model3)
fitMeasures(model3)

sink("EFA_n3.txt")
print(summary(model3))
print(fitMeasures(model3))
sink()

path4 <-'
f1 =~ When.I.work.in.a.group..I.do.higher.quality.work. + My.group.members.help.explain.things.that.I.do.not.understand. + When.I.work.in.a.group..I.am.able.to.share.my.ideas.+The.material.is.easier.to.understand.when.I.work.with.other.students. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful. + My.group.did.higher.quality.work.when.my.group.members.worked.on.tasks.together.
f2 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f4 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
'
model4 <-cfa(path4,data=data)
summary(model4)
fitMeasures(model4)

sink("EFA_n4.txt")
print(summary(model4))
print(fitMeasures(model4))
sink()

path5 <-'
f1 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + The.material.is.easier.to.understand.when.I.work.with.other.students. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful. + My.group.did.higher.quality.work.when.my.group.members.worked.on.tasks.together.
f2 =~ My.group.did.higher.quality.work.when.my.group.members.worked.on.tasks.together.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f4 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f5 =~ The.workload.is.usually.less.when.I.work.with.other.students. + When.I.work.with.other.students.the.work.is.divided.equally.
'
model5 <-cfa(path5,data=data)
summary(model5)
fitMeasures(model5)

sink("EFA_n5.txt")
print(summary(model5))
print(fitMeasures(model5))
sink()

path6 <-'
f1 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f2 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f3 =~ When.I.work.in.a.group..I.do.higher.quality.work. + My.group.members.help.explain.things.that.I.do.not.understand. + The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students.
f4 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + I.become.frustrated.when.my.group.members.do.not.understand.the.material.
f5 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f6 =~ When.I.work.with.other.students.the.work.is.divided.equally.
'
model6 <-cfa(path6,data=data)
summary(model6)
fitMeasures(model6)

sink("EFA_n6.txt")
print(summary(model6))
print(fitMeasures(model6))
sink()

path7 <-'
f1 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f2 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f3 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time.
f4 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f5 =~ The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students.
f6 =~ When.I.work.with.other.students.the.work.is.divided.equally.
'
model7 <-cfa(path7,data=data)
summary(model7)
fitMeasures(model7)

sink("EFA_n7.txt")
print(summary(model7))
print(fitMeasures(model7))
sink()

path8 <-'
f1 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.learn.to.work.with.students.who.are.different.from.me.+I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f2 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f3 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time.
f4 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f5 =~ The.workload.is.usually.less.when.I.work.with.other.students. + When.I.work.with.other.students.the.work.is.divided.equally.
f6 =~ I.try.to.make.sure.my.group.members.learn.the.material.
'
model8 <-cfa(path8,data=data)
summary(model8)
fitMeasures(model8)

sink("EFA_n8.txt")
print(summary(model8))
print(fitMeasures(model8))
sink()

path9 <-'
f1 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f2 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f3 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time.
f4 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f5 =~ The.workload.is.usually.less.when.I.work.with.other.students. + When.I.work.with.other.students.the.work.is.divided.equally.
f6 =~ I.try.to.make.sure.my.group.members.learn.the.material.
f7 =~ My.group.members.help.explain.things.that.I.do.not.understand.
'
model9 <-cfa(path9,data=data)
summary(model9)
fitMeasures(model9)

sink("EFA_n9.txt")
print(summary(model9))
print(fitMeasures(model9))
sink()

