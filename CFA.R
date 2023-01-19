#load libraries
library(lavaan)
library(foreign)

#import data
data <- read.csv(file='ExportedFiles/CFA_full.csv')

path2 <-'
f1 =~ When.I.work.in.a.group..I.do.higher.quality.work. + When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + My.group.members.help.explain.things.that.I.do.not.understand. + The.material.is.easier.to.understand.when.I.work.with.other.students. +  The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + My.group.members.do.not.care.about.my.feelings. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f2 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. +  My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. +  My.group.members.respect.my.opinions. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.learn.to.work.with.students.who.are.different.from.me. + I.also.learn.when.I.teach.the.material.to.my.group.members. + You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
'
model2 <-cfa(path2,data=data)
summary(model2)

path3 <-'
f1 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + I.have.to.work.with.students.who.are.not.as.smart.as.I.am. + When.I.work.with.other.students.the.work.is.divided.equally. + I.become.frustrated.when.my.group.members.do.not.understand.the.material.
f2 =~ My.group.members.help.explain.things.that.I.do.not.understand. + When.I.work.in.a.group..I.am.able.to.share.my.ideas. +  My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. +  My.group.members.do.not.care.about.my.feelings. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
'
model3 <-cfa(path3,data=data)
summary(model3)

path4 <-'
f1 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + My.group.members.help.explain.things.that.I.do.not.understand. +  The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + I.have.to.work.with.students.who.are.not.as.smart.as.I.am. + When.I.work.with.other.students.the.work.is.divided.equally. + I.become.frustrated.when.my.group.members.do.not.understand.the.material.
f2 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f4 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
'
model4 <-cfa(path4,data=data)
summary(model4)

path5 <-'
f1 =~ When.I.work.in.a.group..I.do.higher.quality.work. + When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + My.group.members.help.explain.things.that.I.do.not.understand. +  The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally.
f2 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f4 =~ My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f5 =~ I.have.to.work.with.students.who.are.not.as.smart.as.I.am. +  I.become.frustrated.when.my.group.members.do.not.understand.the.material.
'
model5 <-cfa(path5,data=data)
summary(model5)

path6 <-'
f1 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. +  The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally.
f2 =~ I.feel.I.am.part.of.what.is.going.on.in.the.group. + I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.to.take.on.tasks.that.will.help.me.better.learn.the.material. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f3 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f4 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f5 =~ When.I.work.in.a.group..I.do.higher.quality.work. + The.material.is.easier.to.understand.when.I.work.with.other.students.
f6 =~ I.have.to.work.with.students.who.are.not.as.smart.as.I.am. +  I.become.frustrated.when.my.group.members.do.not.understand.the.material.
'
model6 <-cfa(path6,data=data)
summary(model6)

path7 <-'
f1 =~ When.I.work.in.a.group..I.do.higher.quality.work. + When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally.
f2 =~ When.I.work.in.a.group..I.am.able.to.share.my.ideas. + My.group.members.make.me.feel.that.I.am.not.as.smart.as.they.are. + My.group.members.respect.my.opinions. + I.feel.I.am.part.of.what.is.going.on.in.the.group. + My.group.members.do.not.care.about.my.feelings.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f4 =~ I.learn.to.work.with.students.who.are.different.from.me. + I.also.learn.when.I.teach.the.material.to.my.group.members.
f5 =~ I.prefer.when.the.leadership.role.rotates.between.students. + I.try.to.make.sure.my.group.members.learn.the.material. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful. + My.group.did.higher.quality.work.when.my.group.members.worked.on.tasks.together.
f6 =~ I.have.to.work.with.students.who.are.not.as.smart.as.I.am. +  I.become.frustrated.when.my.group.members.do.not.understand.the.material.
'
model7 <-cfa(path7,data=data)
summary(model7)

path8 <-'
f1 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally.
f2 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f3 =~ I.learn.to.work.with.students.who.are.different.from.me. + I.also.learn.when.I.teach.the.material.to.my.group.members.
f4 =~ I.prefer.when.the.leadership.role.rotates.between.students. + I.try.to.make.sure.my.group.members.learn.the.material. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful. + My.group.did.higher.quality.work.when.my.group.members.worked.on.tasks.together.
f5 =~ I.have.to.work.with.students.who.are.not.as.smart.as.I.am. +  I.become.frustrated.when.my.group.members.do.not.understand.the.material.
f6 =~ My.group.members.help.explain.things.that.I.do.not.understand.
'
model8 <-cfa(path8,data=data)
summary(model8)

path9 <-'
f1 =~ When.I.work.in.a.group..I.end.up.doing.most.of.the.work. + The.work.takes.more.time.to.complete.when.I.work.with.other.students. + My.group.members.help.explain.things.that.I.do.not.understand. + The.material.is.easier.to.understand.when.I.work.with.other.students. + The.workload.is.usually.less.when.I.work.with.other.students. + I.do.not.think.a.group.grade.is.fair. + I.feel.working.in.groups.is.a.waste.of.time. + When.I.work.with.other.students.the.work.is.divided.equally.
f2 =~ I.feel.I.am.part.of.what.is.going.on.in.the.group. +  I.try.to.make.sure.my.group.members.learn.the.material. + I.learn.to.work.with.students.who.are.different.from.me. + I.prefer.when.the.leadership.role.rotates.between.students. + I.also.learn.when.I.teach.the.material.to.my.group.members. + Everyone.s.ideas.are.needed.if.we.are.going.to.be.successful.
f3 =~ You.have.a.certain.amount.of.physics.intelligence..and.you.can.t.really.do.much.to.change.it. + Your.physics.intelligence.is.something.about.you.that.you.can.change. + You.can.learn.new.things..but.you.can.t.really.change.your.basic.physics.intelligence.
f4 =~ My.group.members.help.explain.things.that.I.do.not.understand. + My.group.members.respect.my.opinions. + My.group.members.do.not.care.about.my.feelings.
f5 =~ I.have.to.work.with.students.who.are.not.as.smart.as.I.am. +  I.become.frustrated.when.my.group.members.do.not.understand.the.material.
f6 =~ We.cannot.complete.the.assignment.unless.everyone.contributes.
f7 =~ I.prefer.to.take.on.tasks.that.I.m.already.good.at.
'
model9 <-cfa(path9,data=data)
summary(model9)

