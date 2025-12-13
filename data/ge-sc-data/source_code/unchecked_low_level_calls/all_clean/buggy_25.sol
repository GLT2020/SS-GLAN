pragma solidity ^0.5.11;
contract WhiteBetting {
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
  address payable public owner;
  struct GameInfo {
    uint256 timestamp;
    uint32 odd_homeTeam;
    uint32 odd_drawTeam; 
    uint32 odd_awayTeam;
    uint32 odd_over;
    uint32 odd_under;
    uint32 odd_homeTeamAndDraw;
    uint32 odd_homeAndAwayTeam;
    uint32 odd_awayTeamAndDraw;
    uint8  open_status;
    bool   isDone;
  }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
  mapping(uint64 => GameInfo) public gameList;
  struct BetFixture {
    address payable player;
    uint256 stake;
    uint32  odd;
    uint16  selectedTeam;
  }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
  mapping(uint64 => BetFixture[]) public betList;
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
  event Success(uint256 odd);
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
  event Deposit(address sender, uint256 eth);
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
  event Withdraw(address receiver, uint256 eth);
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
  event NewStake(address player, uint64 fixtureId, uint16 selectedTeam, uint256 stake, uint256 odd );
function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event SetGame(uint64 _fixtureId, uint256 _timestamp, uint32 _odd_homeTeam, uint32 _odd_drawTeam, uint32 _odd_awayTeam, uint32 _odd_over, uint32 _odd_under, uint32 _odd_homeTeamAndDraw, uint32 _odd_homeAndAwayTeam , uint32 _odd_awayTeamAndDraw, uint8 _open_status);
function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event ChangeOdd (uint64 _fixtureId, uint32 _odd_homeTeam, uint32 _odd_drawTeam, uint32 _odd_awayTeam, uint32 _odd_over, uint32 _odd_under, uint32 _odd_homeTeamAndDraw, uint32 _odd_homeAndAwayTeam , uint32 _odd_awayTeamAndDraw);
function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event GivePrizeMoney(uint64 _fixtureId, uint8 _homeDrawAway, uint8 _overUnder);
  constructor() public {
    owner   = msg.sender;
  }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
  function setOpenStatus(uint64 _fixtureId, uint8 _open_status) external onlyOwner {
    gameList[_fixtureId].open_status = _open_status;
  }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
  function changeOdd (uint64 _fixtureId, uint32 _odd_homeTeam, uint32 _odd_drawTeam, uint32 _odd_awayTeam, uint32 _odd_over, uint32 _odd_under, uint32 _odd_homeTeamAndDraw, uint32 _odd_homeAndAwayTeam , uint32 _odd_awayTeamAndDraw ) external onlyOwner {
    gameList[_fixtureId].odd_homeTeam        = _odd_homeTeam;
    gameList[_fixtureId].odd_drawTeam        = _odd_drawTeam;
    gameList[_fixtureId].odd_awayTeam        = _odd_awayTeam;
    gameList[_fixtureId].odd_over            = _odd_over;
    gameList[_fixtureId].odd_under           = _odd_under;
    gameList[_fixtureId].odd_homeTeamAndDraw = _odd_homeTeamAndDraw;
    gameList[_fixtureId].odd_homeAndAwayTeam = _odd_homeAndAwayTeam;
    gameList[_fixtureId].odd_awayTeamAndDraw = _odd_awayTeamAndDraw;
    emit ChangeOdd (_fixtureId, _odd_homeTeam, _odd_drawTeam, _odd_awayTeam, _odd_over, _odd_under, _odd_homeTeamAndDraw, _odd_homeAndAwayTeam , _odd_awayTeamAndDraw);
  }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
  function setGameInfo (uint64 _fixtureId, uint256 _timestamp, uint32 _odd_homeTeam, uint32 _odd_drawTeam, uint32 _odd_awayTeam, uint32 _odd_over, uint32 _odd_under, uint32 _odd_homeTeamAndDraw, uint32 _odd_homeAndAwayTeam , uint32 _odd_awayTeamAndDraw, uint8 _open_status ) external onlyOwner {
    gameList[_fixtureId].timestamp           = _timestamp;
    gameList[_fixtureId].odd_homeTeam        = _odd_homeTeam;
    gameList[_fixtureId].odd_drawTeam        = _odd_drawTeam;
    gameList[_fixtureId].odd_awayTeam        = _odd_awayTeam;
    gameList[_fixtureId].odd_over            = _odd_over;
    gameList[_fixtureId].odd_under           = _odd_under;
    gameList[_fixtureId].odd_homeTeamAndDraw = _odd_homeTeamAndDraw;
    gameList[_fixtureId].odd_homeAndAwayTeam = _odd_homeAndAwayTeam;
    gameList[_fixtureId].odd_awayTeamAndDraw = _odd_awayTeamAndDraw;
    gameList[_fixtureId].open_status         = _open_status;
    gameList[_fixtureId].isDone              = false;
    emit SetGame(_fixtureId, _timestamp, _odd_homeTeam, _odd_drawTeam, _odd_awayTeam, _odd_over, _odd_under, _odd_homeTeamAndDraw, _odd_homeAndAwayTeam , _odd_awayTeamAndDraw, _open_status);
  }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
  function placeBet(uint64 _fixtureId, uint16 _selectedTeam, uint32 _odd) external payable  {
    uint stake = msg.value;
    require(stake >= .001 ether);
    require(_odd != 0 );
    if (_selectedTeam == 1 ) {
      require(gameList[_fixtureId].odd_homeTeam == _odd);
    } else if ( _selectedTeam == 2) {
      require(gameList[_fixtureId].odd_drawTeam == _odd);
    } else if ( _selectedTeam == 3) {
      require(gameList[_fixtureId].odd_awayTeam == _odd);
    } else if ( _selectedTeam == 4) {
      require(gameList[_fixtureId].odd_over == _odd);
    } else if ( _selectedTeam == 5) {
      require(gameList[_fixtureId].odd_under == _odd);
    } else if ( _selectedTeam == 6) {
      require(gameList[_fixtureId].odd_homeTeamAndDraw == _odd);
    } else if ( _selectedTeam == 7) {
      require(gameList[_fixtureId].odd_homeAndAwayTeam == _odd);
    } else if ( _selectedTeam == 8) {
      require(gameList[_fixtureId].odd_awayTeamAndDraw == _odd);
    } else {
      revert();
    }
    require(gameList[_fixtureId].open_status == 3);
    require( now < ( gameList[_fixtureId].timestamp  - 10 minutes ) );
    betList[_fixtureId].push(BetFixture( msg.sender, stake,  _odd, _selectedTeam));
    emit NewStake(msg.sender, _fixtureId, _selectedTeam, stake, _odd );
  }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
  function givePrizeMoney(uint64 _fixtureId, uint8 _homeDrawAway, uint8 _overUnder) external onlyOwner payable {
    require(gameList[_fixtureId].open_status == 3);
    require(gameList[_fixtureId].isDone == false);
    require(betList[_fixtureId][0].player != address(0) );
    for (uint i= 0 ; i < betList[_fixtureId].length; i++){
      uint16 selectedTeam = betList[_fixtureId][i].selectedTeam;
      uint256 returnEth = (betList[_fixtureId][i].stake * betList[_fixtureId][i].odd) / 1000 ;
      if ( (selectedTeam == 1 && _homeDrawAway == 1) 
        || (selectedTeam == 2 && _homeDrawAway == 2) 
        || (selectedTeam == 3 && _homeDrawAway == 3) 
        || (selectedTeam == 4 && _overUnder == 1) 
        || (selectedTeam == 5 && _overUnder == 2) 
        || (selectedTeam == 6 && ( _homeDrawAway == 1 || _homeDrawAway == 2) )
        || (selectedTeam == 7 && ( _homeDrawAway == 1 || _homeDrawAway == 3) )
        || (selectedTeam == 8 && ( _homeDrawAway == 3 || _homeDrawAway == 2) ) 
        ){ 
        betList[_fixtureId][i].player.transfer(returnEth);
      }
    }
    gameList[_fixtureId].open_status = 5;
    gameList[_fixtureId].isDone = true; 
    emit GivePrizeMoney( _fixtureId,  _homeDrawAway,  _overUnder);
  }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
  modifier onlyOwner {
    require (msg.sender == owner, "OnlyOwner methods called by non-owner.");
    _;
  }
  function getBalance() external view returns(uint){
    return address(this).balance;
  }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
  function deposit(uint256 _eth) external payable{
    emit Deposit(msg.sender, _eth);
  }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
  function changeOwner(address payable _newOwner ) external onlyOwner {
    owner = _newOwner;
  }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
  function () external payable{
    owner.transfer(msg.value);    
  }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
  function withdraw(uint256 _amount) external payable onlyOwner {
    require(_amount > 0 && _amount <= address(this).balance );
    owner.transfer(_amount);
    emit Withdraw(owner, _amount);
  }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
}