import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_PATH);
const browser = await chromium.launch({headless: true});
try {
  for (const [name, viewport] of [['desktop', {width: 1440, height: 1000}], ['iphone', {width: 390, height: 844}]]) {
    const page = await browser.newPage({viewport});
    await page.goto('http://127.0.0.1:8517');
    await page.getByRole('button', {name: 'Schedule This Week', exact: true}).click();
    await page.getByRole('button', {name: 'Open Workout', exact: true}).first().click();
    await page.getByRole('heading', {name: 'Lower Body A', exact: true}).scrollIntoViewIfNeeded();
    await page.getByLabel('Weight (lb)', {exact: true}).first().waitFor();
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    assert.equal(await page.locator('[data-testid="stAlertContainer"]').count(), 0);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false, `${name}: page should not overflow horizontally`);
    await page.screenshot({path: `/private/tmp/fitness-workout-${name}.png`});
    await page.getByLabel('Weight (lb)', {exact: true}).first().scrollIntoViewIfNeeded();
    await page.screenshot({path: `/private/tmp/fitness-workout-${name}-sets.png`});
    await page.close();
  }
  console.log('Desktop and iPhone browser checks passed; screenshots are in /private/tmp/fitness-workout-*.png.');
} finally {
  await browser.close();
}
