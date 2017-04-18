import sbt.Keys._
import sbt._
import sbtassembly.AssemblyKeys._

object BuildKaggleTitanic extends Build {
  lazy val id = "kaggle-titanic" // dvergar

  lazy val commonSettings = Seq(
    name := id,
    version := "0.0.1",
    organization := "com.argcv",
    scalaVersion := "2.11.7",
    licenses := Seq("MIT" -> url("http://opensource.org/licenses/MIT")),
    homepage := Some(url("https://github.com/yuikns/kaggle-titanic"))
  )

  lazy val publishSettings = Seq(
    isSnapshot := false,
    publishMavenStyle := true,
    publishTo := {
      val nexus = "https://oss.sonatype.org/"
      if (isSnapshot.value)
        Some("snapshots" at nexus + "content/repositories/snapshots")
      else
        Some("releases" at nexus + "service/local/staging/deploy/maven2")
    },
    publishArtifact in Test := false,
    pomIncludeRepository := { _ => false }
  )

  lazy val dependenciesSettings = Seq(
    resolvers ++= Seq(
      "Atlassian Releases" at "https://maven.atlassian.com/public/",
      "scalaz-bintray" at "https://dl.bintray.com/scalaz/releases",
      Resolver.sonatypeRepo("snapshots"),
      Classpaths.typesafeReleases,
      Classpaths.typesafeSnapshots
    ),
    libraryDependencies ++= Seq(
      "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8", // deeplearning4j
      //"org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8", // deeplearning4j
      //"org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc3.8", // deeplearning4j
      "de.bwaldvogel" % "liblinear" % "1.95", // for liblinear
      "commons-pool" % "commons-pool" % "1.6", // pool for SockPool
      "net.liftweb" % "lift-webkit_2.11" % "3.0-M6", // a light weight framework for web
      "com.google.guava" % "guava" % "18.0", // string process etc. (snake case for example)
      "org.apache.commons" % "commons-csv" % "1.2", // commons, parse csv
      "org.scalatest" % "scalatest_2.11" % "2.2.5" % "test"
    ),
    dependencyOverrides ++= Set(
      "org.scala-lang" % "scala-reflect" % "2.11.7",
      "org.scala-lang" % "scala-compiler" % "2.11.7",
      "org.scala-lang" % "scala-library" % "2.11.7",
      "org.scala-lang.modules" % "scala-xml_2.11" % "1.0.4"
    )
  )

  /**
   * ~~ deduplicate: different file contents found in the following:~~
   * http://stackoverflow.com/questions/25144484/sbt-assembly-deduplication-found-error
   */
  lazy val assemblySettings = Seq(
    assemblyJarName in assembly := s"$id.jar"
  )

  lazy val root = Project(id = id, base = file("."))
    .settings(commonSettings: _*)
    .settings(publishSettings: _*)
    .settings(dependenciesSettings: _*)
    .settings(assemblySettings: _*)
    .dependsOn(valhalla)
    .aggregate(valhalla)

  lazy val valhalla = ProjectRef(file("modules/valhalla"), "valhalla")

}

